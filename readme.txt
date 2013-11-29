团队名称: hadoop_vision
作品名称: 火眼金睛&慧眼识人
队长姓名: 邹晓川
联系方式: zouxiaochuan@163.com

Requriement:
	1. linux-64bit,2.6.18以上
	2. gcc 4.7.2以上

Build:
	进入这个目录，运行make，如无意外make完会显示BUILD SUCCESS.
	
如何测试：
    赛题一（火眼金睛）：
	  进入build/bin目录，运行./testclassification.sh input_pic_dir output_file，
	  其中input_pic_dir是测试图像目录，其中只能有后缀名为.jpg的图像文件
	  output_file为输出文件，每一行的格式为: <filename label>, 每次执行完的文件排列顺序可能会不一样。
	  程序会在当前目录生成一份日志文件log.log用于查找错误。
	赛题二（慧眼识人）：
	  进入build/bin目录，运行./testsegmentation.sh input_pic_dir output_dir,
	  其中input_pic_dir是测试图像目录，其中只能有后缀名为.jpg的图像文件
	  output_dir为输出文件的目录。每个文件的文件名为XXX-profile.jpg
	  程序会在当前目录生成一份日志文件log.log用于查找错误。
	注意事项：
	  1. 对于赛题二，输出的轮廓文件与原始图片一样大。读轮廓文件的时候，需将彩色
	     图转换为灰度图。由于图像保存的时候会有一点误差，所以像素值大于128（255
	     表示白色）的表示背景，像素值小于128（0表示黑色）的表示人物。
      2. 所有的输入输出目录必须要事先存在。
      3. 赛题一和赛题二的日志文件有冲突，所以请分开跑。
      4. 对于单核机器，赛题一处理一张图片大概要4秒钟，赛题二处理一张图片小于2秒
         钟。使用多核机器会成倍的减小处理时间。