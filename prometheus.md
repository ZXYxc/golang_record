### prometheus问题

1、压测并发量，查看pprof的结果

#### 业务量

- 业务指标量
- ingested_samples_per_second：每秒写入的数据点:

  - rate(prometheus_tsdb_head_samples_appended_total[2h])
  - 约为61k，也近似active_time_series / scrape_interval
- Active Series：*A series with an open head chunk* is called an active series

  - prometheus_tsdb_head_series :涵盖过去 1-3 小时（在block为2h时）内存在的每个series
  - count({__name__=~".+"}) ：五分钟内内存中未过时的series
  - 约为410w
  - 频繁变换会导致容易OOM，sum(sum_over_time(scrape_series_added[5m])) by (job)看到。
- bytes_per_sample：每个Sample大小：

  - rate(prometheus_tsdb_compaction_chunk_size_bytes_sum[2h]) /   rate(prometheus_tsdb_compaction_chunk_samples_sum[2h])
  - 约为3.1bytes
  - 官网说：Prometheus 平均每个样本仅存储 1-2 个字节，我们的样本规模是否比较大
- 磁盘占用量：chunks, indexes, tombstones, and various metadata
- 明确需要作为最后展示性能的指标
- 物理磁盘容量：needed_disk_space = retention_time_seconds *ingested_samples_per_second* bytes_per_sample，

  - 每秒钟约产生1.8M的series
  - 我们保留15天（1296000），约产生230GB的chunk数据
- 
- 远程存储 Promscale 资源占用极大，40k samples/s，一天 30 亿，就用掉了将近16 cores/40G 内存/150GB 磁盘。而我们单集群1.50 Mil samples/s，一天就产生 1300 亿左右，而且需求数据双副本，这样的资源需求下如果上了线，仅磁盘单集群 1 天就要耗费 12TB 左右，这样的代价我们是表示有些抗拒的……

#### 指标优化

##### mimirtool

- Grafana 出品了一款工具： mimirtool, 可以通过对比 Prometheus 的指标, 和 AlertManager 以及 Grafana 用到的指标, 来分析哪些指标没有用到。思路：https://isekiro.com/kubernetes%E7%9B%91%E6%8E%A7-prometheus-%E6%8C%87%E6%A0%87%E4%BC%98%E5%8C%96/#mimirtool

#### 查询语句优化

- 避免无意义的二元查询：https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization。vm会将二元表达式中的标签下推到每个查询中去

### 内存存储数据结构

#### 基本结构

##### Samples：基本的存储单元

```golang
type sample struct {
	t int64
	v float64
}
```

##### Labels  []Labels：

- 通过一个Labels(标签们)找到对应的数据了。

![1719305836931](image/prometheus/1719305836931.png)

##### memChunk

- HeadChunk是第一个memChunk，也就是memChunk的数据结构，不同于磁盘中的chunk的概念，是一组Samples的存储方式
- 是一个链表，其中Chunk是个interface，对不同的数据结构有不同的实现

  ```golang
  type memChunk struct {
  	chunk            chunkenc.Chunk
  	minTime, maxTime int64
  	prev             *memChunk // Link to the previous element on the list.
  }


  ```
- 比如以HistogramChunk的结构是 ：

  ```
  //	field →    ts    count zeroCount sum []posbuckets []negbuckets
  //	sample 1   raw   raw   raw       raw []raw        []raw
  //	sample 2   delta xor   xor       xor []xor        []xor
  //	sample >2  dod   xor   xor       xor []xor        []xor
  ```

##### memSeries

- 一个memSeries中有多个memChunk。

```golang
type memSeries stuct {
	......
	ref uint64 // 一个series生成的id
	lst labels.Labels // 对应的标签集合
        meta *metadata.Metadata
	// chunks []*memChunk  旧版表示数据集合
	// headChunk *memChunk 旧版的是只有正在被写入的chunk

	mmappedChunks []*mmappedChunk // 内存映射区块，指向磁盘的不可变数组，按时间戳排序，等待被压缩
	headChunks   *memChunk  //新版的是个链表，会指向全部头部中的headChunk

	......
}
```

- 不同的labels组成了不同的series，有不同的ref。
- ![](https://oscimg.oschina.net/oscnet/up-ff822df5479e5097e07aa91acce21a6d419.png)

#### 为什么内存是两小时

Prometheus将最近的数据保存在内存中，这样查询最近的数据会变得非常快，然后通过一个compactor定时将数据打包到磁盘。数据在内存中最少保留2个小时(storage.tsdb.min-block-duration。至于为什么设置2小时这个值，应该是Gorilla那篇论文中观察得出的结论

![](https://oscimg.oschina.net/oscnet/up-c191096c12ab24d2e0e679446256fbef078.png)

即压缩率在2小时时候达到最高，如果保留的时间更短，就无法最大化的压缩。

#### 如何快速根据label定位memSeries

- 通过Hash的方式，每个LabelSet对应一个Hash值，因此每个Label能够找到若干个memSeries，然后取一个交集
- 由于在Prometheus中会频繁的对map[hash/refId]memSeries进行操作，例如检查这个labelSet对应的memSeries是否存在，不存在则创建等。由于golang的map非线程安全，所以其采用了分段锁去拆分锁。
  ```golang
  // 以分片方式管理序列数据，旨在减少锁竞争并提升并发处理能力。
  // 它包含多个分片，每个分片具有自己的series、哈希表、Samples、lock
  type stripeSeries struct {
      // size indicates the number of stripes in this structure.
      size int

      // 记录refId到memSeries的映射,这里的memSeries不包括Samples，都是Labels
      series []map[chunks.HeadSeriesRef]*memSeries

      // hashes is an array of hashmap structures, used for quickly locating series references.记录hash值到memSeries,hash冲突采用拉链法
      hashes []seriesHashmap

      //  记录refId到Samples的映射
      exemplars []map[chunks.HeadSeriesRef]*exemplar.Exemplar

      // locks is an array of locks, each corresponding to a stripe, used to control concurrent access to stripe data.分段锁
      locks []stripeLock

      // gcMut is a mutex used specifically for garbage collection operations to ensure thread safety.
      gcMut sync.Mutex
  }
  ```
- ![](https://oscimg.oschina.net/oscnet/up-5326b8be04e0d095fc337cc3863f1ef1c1c.png)

##### 倒排索引

假设有下面这样的指标集合，于标签取值不同，我们会有四种不同的memSeries：

```
{__name__:http_requests}{group:canary}{instance:0}{job:api-server}   
{__name__:http_requests}{group:canary}{instance:1}{job:api-server}
{__name__:http_requests}{group:production}{instance:1}{job,api-server}
{__name__:http_requests}{group:production}{instance:0}{job,api-server}
```

memSeries在内存中会有4种，同时内存中还夹杂着其它监控项的series：

![img](https://oscimg.oschina.net/oscnet/up-0065f1f2c4e080313de3bee7e5e83ff17d1.png)

如果没有倒排索引，那么我们必须遍历内存中所有的memSeries(数万乃至数十万)，一一按照Labels去比对,这显然在性能上是不可接受的。而有了倒排索引，我们就可以通过求交集的手段迅速的获取需要哪些memSeries。

![](https://oscimg.oschina.net/oscnet/up-dd6641c51ca9ab93ecaa8f98efc0074bab6.png)

注意，这边倒排索引存储的refId必须是有序的。这样，我们就可以在O(n)复杂度下顺利的算出交集。

- 每次新增一个reries，都要把这个新增的标签的series加入倒排索引

  ```golang

  func (h *Head) getOrCreateWithID(id chunks.HeadSeriesRef, hash uint64, lset labels.Labels) (*memSeries, bool, error) {
  	...//创建series

  	h.metrics.seriesCreated.Inc()
  	h.numSeries.Inc()

  	h.postings.Add(storage.SeriesRef(id), lset)
  	return s, true, nil
  }

  ```

#### 查询流程（简略版)

先查找块（可能是head的在内存中，也可能是磁盘的block），再在索引中找到label的倒排索引，根据倒排索引（label的HashID)找到符合条件的memSeries（refID)，然后根据series中的索引，找到对应的chunk，再组合返回

![1719318012413](image/prometheus/1719318012413.png)

### 磁盘存储数据结构

./data
├── 01BKGV7JBM69T2G1BGBGM6KB12
│   └── meta.json
├── 01BKGTZQ1SYQJTR4PB43C8PD98
│   ├── chunks
│   │   └── 000001
│   ├── tombstones
│   ├── index
│   └── meta.json
├── 01BKGTZQ1HHWHV8FBJXW1Y3W0K

│   └── meta.json
├── 01BKGV7JC0RY8A6MACW02A2PJD
│   ├── chunks
│   │   └── 000001
│   ├── tombstones
│   ├── index
│   └── meta.json
├── chunks_head
│   └── 000001
└── wal
    ├── 000000002
    └── checkpoint.00000001
        └── 00000000

- 通过API做删除时，是软删除，不会立刻清理数据块的内容，而是记录再墓碑中
- 一些概念：

  - [https://www.timescale.com/blog/content/images/2022/06/Figure_1_Prometheus_querying.png](
- 一些prometheus自身监控的关键指标：

  - prometheus_engine_queries：查询处理器处理的请求数量
  - prometheus_engine_query_duration_seconds：单个查询的处理时间
  - prometheus_http_request_duration_seconds_bucket：HTTP请求持续时间的分位数桶计数

  * prometheus_tsdb_head_chunks_created_total是counter类型的指标，其值会一直增加，含义是[时序数据库](https://cloud.tencent.com/product/ctsdb?from_column=20065&from=20065)tsdb的head中创建的chunk数量
  * https://blog.csdn.net/weixin_43757402/article/details/129302560
  * **go_memstats_alloc_bytes** – 该指标展示了在 [堆](https://en.wikipedia.org/wiki/Memory_management#HEAP) 上为对象分配了多少字节的内存。该值与 **go_memstats_heap_alloc_bytes** 相同。该指标包括所有可达（reachable）堆对象和不可达(unreachable)对象(GC尚未释放的）占用的内存大小。
- 存储逻辑：https://www.slideshare.net/slideshow/how-prometheus-store-the-data/243217570 （prometheus交流大会PPT)
- 查询工作原理：https://www.timescale.com/blog/how-prometheus-querying-works-and-why-you-should-care/

#### prometheus存储Record

##### meta.json

我们可以通过检查meta.json来得到当前Block的一些元信息，该Block是由31个原始Block经历5次压缩而来。最后一次压缩的三个Block ulid记录在parents中。如下图所示:

```
"ulid":"01EXTEH5JA3QCQB0PXHAPP999D",
	// maxTime - maxTime =>162h
	"minTime":1610964800000,
	"maxTime":1611548000000
	......
	"compaction":{
		"level": 5,
		"sources: [
			31个01EX......
		]
	},
	"parents: [
		{
			"ulid": 01EXTEH5JA3QCQB1PXHAPP999D
			...
		}
		{
			"ulid": 01EXTEH6JA3QCQB1PXHAPP999D
			...
		}
				{
			"ulid": 01EXTEH5JA31CQB1PXHAPP999D
			...
		}
	]
```

##### Series

- sereis会产生独一无二的序列号ID，用于查找
- series会记录全部label，同样的label的series只会在head中被记录一次

##### Sample

- samples会记录所属的series的ID，为了确保每个sample都能找到所属的series，会先写入series再写入sample
- The Samples record is written for all write requests that contain a sample.

##### WAL

- Head中先创建sereis，再写入record，samples是先写入record，再在head中创建
- 先创建墓碑，再适时删除
- WAL的删除发生在块压缩时
- 先写入WAL，再执行
- 因为series是一次写入，为了避免被频繁删除，每次只删除2/3的segments.，并留下一个checkpoint
  - 所选文本中讨论了在Prometheus的TSDB中，对于那些不再在Head Block中的系列记录，以及在特定时间点T之前的所有样本和墓碑记录如何处理。具体包括以下几点：
    1. 删除所有不再在Head Block中的系列记录。
    2. 删除所有在时间点T之前的样本。
    3. 删除所有在时间点T之前的墓碑记录。
    4. 保留剩余的系列、样本和墓碑记录，按照它们在WAL中出现的顺序保持不变。
    5. checkpoint.X where X is the last segment number on which the checkpoint was being created ，If there were any older checkpoints, they are deleted at this point.
  - 代码引用
    - WAL（Write-Ahead-Log）的实现和逻辑分布信息。在tsdb/wal/wal.go文件中，包含了处理字节记录、进行低级磁盘交互的WAL实现。该文件负责编写字节记录并迭代记录（同样以字节片段形式）。tsdb/record/record.go文件则包含了具有其编码和解码逻辑的各种记录。而关于检查点逻辑则在tsdb/wal/checkpoint.go文件中。tsdb/head.go文件涵盖了创建和编码记录、调用WAL写入，调用检查点和WAL截断，以及重放WAL记录、解码它们并恢复内存状态的剩余部分。

##### 磁盘读取全流程：

https://www.cnblogs.com/alchemystar/p/14462052.html

#### prometheus的mmap

- 2.19开始使用mmap技术。新特性：https://grafana.com/blog/2020/06/10/new-in-prometheus-v2.19.0-memory-mapping-of-full-chunks-of-the-head-block-reduces-memory-usage-by-as-much-as-40/

#### chunks

##### CUT文件切分

所有的Chunk文件在磁盘上都不会大于512M,对应的源码为:

```golang
func (w *Writer) WriteChunks(chks ...Meta) error {
	......
	for i, chk := range chks {
		cutNewBatch := (i != 0) && (batchSize+SegmentHeaderSize > w.segmentSize)
		......
		if cutNewBatch {
			......
		}
		......
	}
}
```

一个Chunks文件包含了非常多的内存Chunk结构,如下图所示:

![](https://oscimg.oschina.net/oscnet/up-33fd81c0c76bf5af37c618ac5b6ede73ee4.png)

我们怎么在chunks中找到series对应Chunk的。

在index中，通过series记录的chunks的文件名(000001，前32位)以及(offset,后32位)编码到一个int类型的refId中，使得我们可以轻松的通过这个id获取到对应的chunk块。

##### chunks文件通过mmap去访问

由于chunks文件大小基本固定(最大512M),所以我们很容易的可以通过mmap去访问对应的数据。直接将对应文件的读操作交给操作系统，既省心又省力。对应代码为:

```golang
func NewDirReader(dir string, pool chunkenc.Pool) (*Reader, error) {
	......
	for _, fn := range files {
		f, err := fileutil.OpenMmapFile(fn)
		......
	}
	......
	bs = append(bs, realByteSlice(f.Bytes()))
}
通过sgmBytes := s.bs[offset]就直接能获取对应的数据
```


#### index索引结构

前面介绍完chunk文件，我们就可以开始阐述最复杂的索引结构了。

##### 找到block

索引就是为了让我们快速的找到想要的内容，为了便于理解，这里通过一次数据的寻址来探究Prometheus的磁盘索引结构。考虑查询一个拥有三个标签的series的查询：

```css

({__name__:http_requests}{job:api-server}{instance:0})
且时间为start/end的所有序列数据
```

我们先从选择Block开始,遍历所有Block的meta.json，根据start/end找到具体的Block
![img](https://oscimg.oschina.net/oscnet/up-3de2e009c02e355e03c5240cdf56020b324.png)
前文说了，通过Labels找数据是通过倒排索引。我们的倒排索引是保存在index文件里面的。 那么怎么在这个单一文件里找到倒排索引的位置呢？这就引入了TOC(Table Of Content)

##### 找到倒排索引表的位置，TOC(Table Of Content)

![](https://oscimg.oschina.net/oscnet/up-c3d3a625a7b2e0de94d24f5dd53405f2d28.png)
由于index文件一旦形成之后就不再会改变，所以Prometheus也依旧使用mmap来进行操作。采用mmap读取TOC非常容易:

```go
func NewTOCFromByteSlice(bs ByteSlice) (*TOC, error) {
	......
	// indexTOCLen = 6*8+4 = 52
	b := bs.Range(bs.Len()-indexTOCLen, bs.Len())
	......
	return &TOC{
		Symbols:           d.Be64(),
		Series:            d.Be64(),
		LabelIndices:      d.Be64(),
		LabelIndicesTable: d.Be64(),
		Postings:          d.Be64(),
		PostingsTable:     d.Be64(),
	}, nil
}
```

##### 找到对应的倒排索引的位置--Posting offset table

首先我们访问的是**Posting offset table**。由于倒排索引按照不同的LabelPair(key/value)会有非常多的条目。所以Posing offset table就是决定到底访问哪一条Posting索引。offset就是指的这一Posting条目在文件中的偏移。
![](https://oscimg.oschina.net/oscnet/up-0b7a401836c5b8e549501f9ab6fc2161025.png)

##### 根据Posting倒排索引找到对应的Series

我们通过三条Postings倒排索引索引取交集得出

```undefined
{series1,Series2,Series3,Series4}
∩
{series1,Series2,Series3}
∩
{Series2,Series3}
=
{Series2,Series3}
```

也就是要读取Series2和Serie3中的数据，而Posting中的Ref(Series2)和Ref(Series3)即为这两Series在index文件中的偏移。
![](https://oscimg.oschina.net/oscnet/up-ede75bbb021a5450c9d1a18bb8f7bdf0202.png)
Series以Delta的形式记录了chunkId以及该chunk包含的时间范围。

##### 根据series查找chunk

这样就可以很容易过滤出我们需要的chunk,然后再按照chunk文件的访问，即可找到最终的原始数据。

##### SymbolTable

值得注意的是，为了尽量减少我们文件的大小，对于Label的Name和Value这些有限的数据，我们会按照字母序存在符号表中。由于是有序的，所以我们可以直接将符号表认为是一个
[]string切片。然后通过切片的下标去获取对应的sting。考虑如下符号表:
![](https://oscimg.oschina.net/oscnet/up-7112b989bd70a6c6f785c07540d1276a43b.png)
读取index文件时候，会将SymbolTable全部加载到内存中，并组织成symbols []string这样的切片形式，这样一个Series中的所有标签值即可通过切片下标访问得到。

##### Label Index以及Label Table

事实上，前面的介绍已经将一个普通数据寻址的过程全部讲完了。但是index文件中还包含label索引以及label Table，这两个是用来记录一个Label下面所有可能的值而存在的。
这样，在正则的时候就可以非常容易的找到我们需要哪些LabelPair。详情可以见前篇。
![](https://oscimg.oschina.net/oscnet/up-c704fa6a744fcb356a150bc95ca73bc3333.png)

事实上，真正的Label Index比图中要复杂一点。它设计成一条LabelIndex可以表示(多个标签组合)的所有数据。不过在Prometheus代码中只会采用存储一个标签对应所有值的形式。

##### 完整的index文件结构

这里直接给出完整的index文件结构，摘自Prometheus中index.md文档。

```css
┌────────────────────────────┬─────────────────────┐
│ magic(0xBAAAD700) <4b>     │ version(1) <1 byte> │
├────────────────────────────┴─────────────────────┤
│ ┌──────────────────────────────────────────────┐ │
│ │                 Symbol Table                 │ │
│ ├──────────────────────────────────────────────┤ │
│ │                    Series                    │ │
│ ├──────────────────────────────────────────────┤ │
│ │                 Label Index 1                │ │
│ ├──────────────────────────────────────────────┤ │
│ │                      ...                     │ │
│ ├──────────────────────────────────────────────┤ │
│ │                 Label Index N                │ │
│ ├──────────────────────────────────────────────┤ │
│ │                   Postings 1                 │ │
│ ├──────────────────────────────────────────────┤ │
│ │                      ...                     │ │
│ ├──────────────────────────────────────────────┤ │
│ │                   Postings N                 │ │
│ ├──────────────────────────────────────────────┤ │
│ │               Label Index Table              │ │
│ ├──────────────────────────────────────────────┤ │
│ │                 Postings Table               │ │
│ ├──────────────────────────────────────────────┤ │
│ │                      TOC                     │ │
│ └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## tombstones

由于Prometheus Block的数据一般在写完后就不会变动。如果要删除部分数据，就只能记录一下删除数据的范围，由下一次compactor组成新block的时候删除。而记录这些信息的文件即是tomstones。

## 总结

Prometheus作为时序数据库，设计了各种文件结构来保存海量的监控数据，同时还兼顾了性能。只有彻底了解其存储结构，才能更好的指导我们应用它！

### 使用了VictoriaMetircs的公司

- AfterShip：[https://mp.weixin.qq.com/s/px4Zw_LDn6cHD3y1--GUAQ](https://mp.weixin.qq.com/s/px4Zw_LDn6cHD3y1--GUAQ)
  - [https://blog.csdn.net/kinRongG/article/details/135532690](https://blog.csdn.net/kinRongG/article/details/135532690)
- 网易互娱：[https://www.51cto.com/article/718085.html](https://www.51cto.com/article/718085.html)
- 道客云 ： [https://mp.weixin.qq.com/s/GdNq4tHdG4J2NnqI_atEjA](https://mp.weixin.qq.com/s/GdNq4tHdG4J2NnqI_atEjA)
- VIVO： [https://www.cnblogs.com/vivotech/p/17637510.html](https://www.cnblogs.com/vivotech/p/17637510.html)
  - 曾经单集群若干prometheus
- 小红书： [https://xie.infoq.cn/article/f7ce46d8a02df660fb6ece63f](https://xie.infoq.cn/article/f7ce46d8a02df660fb6ece63f)
  - 曾经单集群若干prometheus
  - __我们将 Kube-state-metrics 调整为 Statefulset 的分片部署方式__
  - 计算下推到存储节点（预查询
  - 查询数量限制
- 滴滴： [https://cloud.tencent.com/developer/article/2367238](https://cloud.tencent.com/developer/article/2367238)
- 新东方： [https://cloud.tencent.com/developer/article/2384359](https://cloud.tencent.com/developer/article/2384359)
- 携程：[https://cloud.tencent.com/developer/article/1950680](https://cloud.tencent.com/developer/article/1950680)
  - [https://cloud.tencent.com/developer/article/2082105](https://cloud.tencent.com/developer/article/2082105)
  - 配置中心下发预聚合
- 得物：[https://mp.weixin.qq.com/s?__biz=MzIzNjUxMzk2NQ==&amp;mid=2247532886&amp;idx=1&amp;sn=965046963d8e50fae57743dbdff103f5&amp;scene=21#wechat_redirect](https://mp.weixin.qq.com/s?__biz=MzIzNjUxMzk2NQ==&mid=2247532886&idx=1&sn=965046963d8e50fae57743dbdff103f5&scene=21#wechat_redirect)
  - 同比、环比聚合
  - trace聚合
- 货拉拉：[https://zhuanlan.zhihu.com/p/611246151](https://zhuanlan.zhihu.com/p/611246151)
- 去哪儿：[https://dbaplus.cn/news-72-5174-1.html](https://dbaplus.cn/news-72-5174-1.html)
  - 全切vms
- 网易云音乐： [https://juejin.cn/post/7322268449409744931](https://juejin.cn/post/7322268449409744931)
- shoppee 存储用vm
- 工行 存储用vm

### 原生prometheus优化方案

- 百度： [https://developer.baidu.com/article/detail.html?id=294702](https://developer.baidu.com/article/detail.html?id=294702)
  - 自研了 PromAgent 与 PromScheduler 模块，PromAgent实时感知 Prometheus 采集压力的变化，并上报给 PromScheduler；PromScheduler 收到采集压力数据后，其分片管理功能根据自动分配算法进行自动伸展
  - 自研流式处理做流式计算动态预查询
  - 自研存储降低采样
- Cloudflare: [https://www.infoq.cn/article/QFFPRKN7a6y3L1Q7geIh](https://www.infoq.cn/article/QFFPRKN7a6y3L1Q7geIh)
  - 包含了prometheus如何存入tsDB的，以及何时从内存中丢失
  - sample_limit限制单个metrics的sample（同一个时间点的时间序列）上限
- 饿了么：[https://mp.weixin.qq.com/s?__biz=MzU4NzU0MDIzOQ==&amp;mid=2247494222&amp;idx=1&amp;sn=143a8ad9e4da4bdf9e7a6c4c738e3bf2&amp;scene=21#wechat_redirect](https://mp.weixin.qq.com/s?__biz=MzU4NzU0MDIzOQ==&mid=2247494222&idx=1&sn=143a8ad9e4da4bdf9e7a6c4c738e3bf2&scene=21#wechat_redirect)
  - 对接自研中间件
- 脉脉：没用prometheus 使用了clickhourse，自己做分布式上传

### Prometheus可优化方法

- 指标优化

  - 对多余指标relabel，比如本来有pod:namespace，取一个新字段id记录二者，再丢掉本来的pod和namespace
  - relabel_config 发生在采集之前，metric_relabel_configs 发生在采集之后，合理搭配可以满足场景的配置
  - 
- prometheus负载均衡

  - prometheus水平扩容还是多实例的prometheus
- 高可用副本数据同步
  ------------------

##### 介绍VictoriaMetrics

- 如何优化了时序数据库。

优点：

- __根据容器可用的 CPU 数量计算协程数量__
- __区分 IO 协程和计算协程，同时提供了协程优先级策略__
- __使用 ZSTD 压缩传输内容降低磁盘性能要求__
- __根据可用物理内存限制对象的总量，避免 OOM__
- __区分 fast path 和 slow path，优化 fast path 避免 GC 压力过大__
