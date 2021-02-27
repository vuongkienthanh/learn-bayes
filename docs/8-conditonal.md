---
title: "Chapter 8: Conditional Manatees"
description: "Chương 8: Những con lợn biển có điều kiện"
---


- [8.1 Xây dựng sự tương tác](#a1)
- [8.2 Tính đối xứng của tương tác](#a2)
- [8.3 Tương tác liên tục](#a3)
- [8.4 Tổng kết](#a4)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation
az.style.use("fivethirtyeight"){% endhighlight %}

Lợn biển (manatee - *Trichechus manatus*) là một động vật hữu nhũ chậm chạp, sống dưới vùng nước ấm và nông. Lợn biển không có thiên dịch tự nhiên, nhưng chúng có chia sẽ nước với những chiếc thuyền máy. Và thuyền máu có cánh quạt. Khi những con lợn biển liên quan với loài voi và có lớp da rất dày, cánh quạt có thể giết chúng. Phần lớn các con lợn biển trưởng thành đều có sẹo từ va chạm với thuyền ([**HÌNH 8.1, trên**](#f1))

<a name="f1"></a>![](/assets/images/fig 8-1.png)
<details class="fig"><summary>Hình 8.1: TRÊN: Các vết sẹo trên lưng của 5 con lợn biển Florida. Các dãy trầy xước, như trên các cá thể Africa và Flash, là dấu hiệu của vết thương do cánh quạt. DƯỚI: Ba ví dụ thiệt hại trên những chiếc máy bay thả bom A.W.38 sau nhiệm vụ.</summary></details>

Armstrong Whitworth A.W.38 Whitley là máy bay thả bom tiền tuyến của Lực Lượng Hàng Không Hoàng Gia. Vào Thế Chiến II, A.W.38 mang bom và thả vào địa phận của Đức. Không giống như lợn biển, A.W.38 có thiên địch tự nhiên hung tợn: pháo phản không. Nhiều máy bay không bao giờ trở về từ nhiệm vụ. Và thiệt hại trên những chiếc còn sống sót đã chứng minh điều đó ([**HÌNH 8.1, dưới**](#f1))

Lợn biển có giống máy bay A.W.38 không? Trong cả hai trường hợp - vết trầy do máy quạt của lợn biển và các lỗ thủng do đạn của máy bay - chúng ta muốn làm gì đó để cải thiện khả năng giúp lợn biển và máy bay sống sót. Nhiều người quan sát nghĩ rằng giúp lợn biển và máy bay nghĩa là giảm loại tổn thương mà chúng ta nhìn thấy trên chúng. Với lợn biển, nó nghĩa là cần thêm lớp bảo vệ trên cánh quạt (của thuyền, không phải của lợn biển). Với máy bay, nó nghĩa là cần thêm giáp ở những bộ phận máy bay có nhiều tổn thương nhất.

Trong cả hai trường hợp, bằng chứng đã gây hiểu sai. Cánh quạt không gây ra những tổn thương và cái chết cho lợn biển. Thực vậy tử thiết trên chúng khẳng định sự va chạm với các bộ phận tù của chiếc thuyền nhưng mái chéo, cho tổn thương nhiều hơn. Tương tự, tăng giáp ở những thành phần tổn thương không cải thiện cho các máy bay thả bom. Thay vào đó, nâng cấp máy bay thả bom A.W.38 nghĩa là nâng giáp các bộ phận *không bị tổn thương*. Ví dụ từ những con lợn biển và máy bay thả bom còn sống sót là gây hiểu sai, bởi vì nó *đặt điều kiện* trên sống còn. Lợn biển và máy bay bị tiêu diệt trông sẽ khác. Một con lợn biển bị va chạm với máu chèo sẽ ít khả năng sống sót hơn nhưng con bị cánh quạt trợt qua. Nên giữa các con sống sót, vết trầy do cánh quạt là thường gặp. Tương tự, những máy bay trở về được không có vết thương rõ ràng trên động cơ và buồng lái. Họ may mắn. Máy bay thả bom không về nhà được thì ít như vậy hơn. Để có được đáp án chính xác, trong hai trường hợp trên, chúng ta cần phải nhận ra loại tổn thương nhìn thấy này là được đặt điều kiện trên sống còn.

**ĐẶT ĐIỀU KIỆN** là một trong những nguyên tắc quan trọng nhất trong suy luận thống kê. Data, như các vết xước lợn biển và tổn thương máy bay, được đặt điều kiện trên cách chúng được đưa vào mẫu. Phân phối posterior được đặt điều kiện trên data. Tất cả mọi suy luận bằng mô hình được đặt điều kiện trên mô hình. Mọi suy luận được đặt điều kiện trên một thứ gì đó, cho dù chúng ta có nhận ra hay không.

Và một phần lớn sức mạnh của mô hình thống kê đến từ việc tạo ra các thiết bị cho phép xác suất được đặt điều kiện trên các khía cạnh của từng trường hợp. Mô hình tuyến tính mà bạn quen thuộc mà thiết bị thô sơ cho phép mọi kết cục $y_i$ được đặt điều kiện trên một tập các biến dự đoán cho mỗi trường hợp $i$. Giống như các epicycle của mô hình Ptolemaic và Kopernikan (Chương 4 và 7), mô hình tuyến tính cho chúng ta phương pháp mô tả khả năng đặt điều kiện.

Mô hình tuyến tính đơn giản thông thường không có khả năng cung cấp đủ các phép đặt điều kiện, tuy nhiên . Mọi mô hình đến bây giờ trong sách này giả định mỗi biến dự đoán có mối quan hệ độc lập với trung bình của kết cục. Nhưng nếu chúng ta muốn cho phép mối quan hệ được đặt điều kiện thì sao? Ví dụ, trong data sữa các loài khỉ từ các chương trước, giả sử quan hệ giữa năng lượng sữa và kích thước não thay đổi theo nhóm loài (khỉ, vượn, tinh tinh). Điều này giống như nói rằng ảnh hưởng của kích thước não trên năng lượng sữa được đặt điều kiện trên nhóm loài. Mô hình tuyến tính từ chương trước không giải quyết được câu hỏi này.

Để mô hình hoá điều kiện sâu hơn - khi mức độ quan trọng của một biến phụ thuộc vào một biến dự đoán khác - chúng ta cần **SỰ TƯƠNG TÁC (INTERACTION)** (cũng được biết **SỰ ĐIỀU TIẾT - MODERATION**). Tương tác là một loại đặt điều kiện, một cách để cho phép tham số (thực ra là phân phối posterior) được đặt điều kiện trên các khía cạnh xa hơn của data. Loại tương tác đơn giản nhất, tương tác tuyến tính, được xây dựng bằng cách mở rộng chiến thuật mô hình tuyến tính vào tham số trong mô hình tuyến tính. Cho nên nó đồng nghĩa với việc thay thế epicycle trên epicycle trong mô hình Ptolemaic và Kopernikan. Nó mang tính mô tả, nhưng rất mạnh.

Tổng quát hơn, tương tác là trung tâm của đa số mô hình thống kê đằng sau thế giới ấm áp của kết cục Gaussian và mô hình tuyến tính của trung bình. Trong mô hình tuyến tính tổng quát (GLM, Chương 10 và sau đó), ngay cả khi người ta không định nghĩa rõ ràng các biến là tương tác, chúng vẫn tương tác ở một mức độ nào đó. Mô hình đa tầng cũng cho hiệu ứng tương tự. Mô hình đa tầng thông thường là một mô hình tương tác khổng lồ, trong đó các giá trị ước lượng (intercept và slope) được đặt điều kiện cho cụm (người, loài, làng, thành phố, vũ trụ) trong data. Hiệu ứng tương tác đa tầng là phức tạp. Chúng không chỉ cho phép ảnh hưởng của biến dự đoán thay đổi phụ thuộc vào một biến khác, mà còn ước lượng khía cạnh của *phân phối* của những thay đổi đó. Điều này nghe có vẻ thiên tài, hoặc điên rồ, hoặc cả hai. Cho dù thế nào, bạn không thể có sức mạnh của mô hình tầng mà không có nó.

Mô hình cho phép tương tác phức tạp thì dễ fit vào data. Nhưng chúng cũng được cho là khó hiểu hơn. Và nên tôi dành chương này nói về các hiệu ứng tương tác đơn giản: làm sao để định nghĩa, diễn giải, và minh hoạ chúng. Chương này bắt đầu bằng một trường hợp tương tác giữa một biến phân nhóm và một biến liên tục. Trong bối cảnh này, rất dễ để nhận ra dạng giả thuyết cho phép sự tương tác. Rồi sau đó chương này nói về tương tác phức tạp hơn giữa các biến dự đoán liên tục. Nó khó hơn. Trong tất cả các phần của chương này, dự đoán của mô hình được minh hoạ, trung bình hoá trên tính bất định trong tham số.

Sự tương tác là bình thường, nhưng chúng không dễ. Hi vọng là chương này là tạo một nền tảng vững chắc cho việc diễn giải mô hình tuyến tính tổng quát và mô hình đa tầng trong các chương sau.

<div class="alert alert-info">
<p><strong>Minh tinh thống kê, Abraham Wald.</strong> Câu chuyện máy bay thả bom trong Thế Chiến II là tác phẩm của Abraham Wald (1902-1950). Wald sinh ra ở nơi mà bây giờ gọi là Romania, nhưng di cư sang Mỹ sau khi Nazi xâm chiếm nước Áo. Wald đã cống hiến rất nhiều trong cuộc đời ngắn ngủi của ông. Có lẽ công tình liên quan nhất đến tài liệu này, là Wald đã chứng mình rằng với nhiều loại quy luật để quyết định theo thống kê, luôn luôn tồn tại một quy luật Bayes chí ít tốt bằng nhiều quy luật non-Bayes. Wald đã chứng minh điều này, một cách xuất sắc, bắt đầu với các tiền đề non-Bayes, và nên phe anti-Bayes không thể mặc kệ nó nữa. Công trình này được tóm tắc trong sách 1950 của Wald, được phát hành chỉ trước khi ông mất. Wald chết khi quá trẻ, từ một vụ rơi máy bay khi tham quan Ấn Độ.</p></div>

## <center></center>8.1 Xây dựng sự tương tác<a name="a1"></a>
## <center></center>8.2 Tính đối xứng của tương tác<a name="a2"></a>
## <center></center>8.3 Tương tác liên tục<a name="a3"></a>
## <center></center>8.4 Tổng kết<a name="a4"></a>


<a name="f1"></a>![](/assets/images/fig -.svg)
<details class="fig"><summary></summary>
{% highlight python %}{% endhighlight %}</details>

---

<details><summary>Endnotes</summary>
<ol start="" class='endnotes'>
    <li></li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
</details>