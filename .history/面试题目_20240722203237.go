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
}

funct _append(a int){
	a = append(a,6)
}