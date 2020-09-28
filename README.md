# LR-DeepFM

1. 数据文件为frappe.train2.libfm/frappe.test2.libgm/frappe.validation2.libfm，均为原数据文件通过data_processing.py转化而来，主要思路是去掉了每个数据中的":1"，用后一个数减去前一个数，使得数据更有实际意义
2. Frappe_new.train2.csv/Frappe_new.test2.csv/Frappe_new.validation2.csv是把数据文件读入CSV的结果，从而便于对数据进行处理
3. LR.py是调用sklearn包写的逻辑斯蒂回归方法，准确率约为83%
4. LR_pytorch.py是用pytorch写的逻辑斯蒂回归方法，准确率仅为33.4%
4. DeepFM.py是根据DeepFM原理直接写的DeepFM方法，准确率约为85%
5. DeepFM_pytorch.py是用pytorch写的DeepFM方法，不过仍存在一些问题，目前准确率仅为33%





