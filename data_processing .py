def processing(file_in ,file_out): 
    # 打开文件
    fo = open(file_in, "r+")
    fp = open(file_out, "r+")
    print ("文件名为: ", fo.name)


    # line = fo.readline()
    # print ("读取第一行 %s" % (line))
    for line in fo.readlines():   
        #print ("读取第一行 %s" % (line))
        p = line.split()
        #print(p[1],p[2])
        #print(len(p))
        for i in range(10) :
            p[i+1] = p[i+1][:-2]
            p[i+1] = int(p[i+1])
        for i in range(10) :
            if i == 9:
                break
            p[10-i] = p[10-i] - p[9-i] 
        # line = fo.readline(5)
        # print ("读取的字符串为: %s" % (line))
        #print(p)
        str_p=[str(x) for x in p]
        st = " " 
        seq = st.join(str_p)
        #print(seq)
        fp.write(seq + '\n')
        #fp.write("hello world")
    # 关闭文件
    fo.close()
    fp.close()

processing("frappe.validation1.libfm","frappe.validation2.libfm")
