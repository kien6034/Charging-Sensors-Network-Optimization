Mô tả dữ liệu thực nghiệm
1) Tọa độ trạm cơ sở và tọa độ của các cảm biến được sinh mô phỏng sử dụng phần mềm wissim
2) Cảm biến được rải theo các phân phối: lưới (bộ có ký hiệu là g), đều (bộ có ký hiệu là u), log chuẩn (bộ có ký hiệu n)
3) Bộ small_network có diện tích 500mx500m
   Bộ large_network có diện tích 1000m x 1000m
4) Trong mỗi bộ small_network và large_network gồm hai thư mục
   thứ nhất là "topodata" lưu các tệp .txt gồm tọa độ trạm và tọa độ các cảm biến. Các bạn chạy mô phỏng mạng sinh ra công suất tiêu thụ và công suất tiêu thụ trung bình 
   chỉ quan tâm tới thư mục "topodata"
   thứ hai là "visualization" là các tệp ".png" là hiển thị cấu trúc mạng theo các tệp ".txt" trong thư mục "topodata"
5) Trong thư mục topodata gồm có ba bộ dữ liệu về phân phối của các cảm biến (lưới - grid, đều - uniform và log chuẩn -normal)
	Trong mỗi phân phối: tọa độ trạm cơ sở sẽ được đặt ở 5 vị trí: (0,0) (0,500), (500, 0), (500,500) và (250,250)
6) Mỗi tệp ".txt" trong mỗi thư mục con của topodata chứa thông tin về: phân phối và số lượng cảm biến
	ví dụ:
	g_20 => g là phân phối cảm biến trên các ô lưới, 20 là số lượng cảm biến trong mạng
	Nội dung mỗi tệp ".txt" chứa n+1 dòng (trong đó n là số lượng cảm biến)
	- Dòng đầu tiên là tọa độ trạm cơ sở
	- n dòng tiếp theo là các thông tin của n cảm biến bao gồm: tọa độ ox, tọa độ oy, công suất tiêu thụ trung bình trong 10000(s) chạy mô phỏng thật, năng lượng còn lại của cảm biến
	
	* Tham số mạng:
	
	+ Tham số của cảm biến:
		E_max = 10800 (J): dung lượng pin tối đa
		E_min = 540(J): dung lượng pin tối thiểu để hoạt động
	+ Tham số của xe sạc (thiết bị sạc)
		E_MC = 108000 (J) : Dung lượng tối đa của bộ sạc
		U = 5(J/s): Năng lượng mà xe sạc cho cảm biến trong 1 giây = năng lượng nhận của cảm biến trong 1 (s) => bảo toàn năng lượng
		P_M = 1 (J/s): Năng lượng mà xe tiêu hao trong 1 giây cho quá trình di chuyển
		
		alpha= 3600 (tham số alpha và beta dùng cho mô hình sạc đa điểm thay cho tham số U)
		beta = 30
		
		