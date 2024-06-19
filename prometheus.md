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

### 存储数据结构

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

  - [https://www.timescale.com/blog/content/images/2022/06/Figure_1_Prometheus_querying.png]()
- 一些prometheus自身监控的关键指标：

  - prometheus_engine_queries：查询处理器处理的请求数量
  - prometheus_engine_query_duration_seconds：单个查询的处理时间
  - prometheus_http_request_duration_seconds_bucket：HTTP请求持续时间的分位数桶计数

  * prometheus_tsdb_head_chunks_created_total是counter类型的指标，其值会一直增加，含义是[时序数据库](https://cloud.tencent.com/product/ctsdb?from_column=20065&from=20065)tsdb的head中创建的chunk数量
  * https://blog.csdn.net/weixin_43757402/article/details/129302560
  * **go_memstats_alloc_bytes** – 该指标展示了在 [堆](https://en.wikipedia.org/wiki/Memory_management#HEAP) 上为对象分配了多少字节的内存。该值与 **go_memstats_heap_alloc_bytes** 相同。该指标包括所有可达（reachable）堆对象和不可达(unreachable)对象(GC尚未释放的）占用的内存大小。
- 存储逻辑：https://www.slideshare.net/slideshow/how-prometheus-store-the-data/243217570 （prometheus交流大会PPT)
- 查询工作原理：https://www.timescale.com/blog/how-prometheus-querying-works-and-why-you-should-care/

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
