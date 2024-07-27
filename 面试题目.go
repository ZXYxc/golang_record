func test1()  {
	 t := make(map[string]string)
	go func(){
		t["a"] = ""
	}
		go func(){
		t["ab"] = ""
	}
}

func test2()  {
	a := make([]int,0,10)
	a = []int{1,2,3,4,5}
	_append(a)

	fmt.print(a)
}

func _append(a []int){
	a = append(a,6)
}

短链系统：www.bilibli/com/video/BV9j7xh92c?q=fh2n94f2&mod=n2f3nf  ----> b23.tv/fm92f2

        
1、短链如何生成
2、数据存储和数据存储设计
3、短链访问的效果如何等同于长链的效果


|id|short_url|long_url|ctime|mtime|

id ---> 100000 -(16进制)-> 63 a-zA-Z0-9

redis key(short)val(long_url)  