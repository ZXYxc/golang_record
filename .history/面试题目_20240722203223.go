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
	a := make([]int,)
}

funct _append(a int){
	a = append(a,6)
}