---
title: "Chapter 4: Geocentric Models"
description: "Chương 4: Mô hình địa tâm"
---

- [4.1 Tại sao phân phối normal lại normal](#a1)
- [4.2 Ngôn ngữ mô tả model](#a2)
- [4.3 Model Gaussian chiều cao](#a3)
- [4.4 Linear Model](#a4)
- [4.5 Đường cong từ đường thẳng](#a5)
- [4.6 Tổng kết](#a6)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline
from jax import vmap, random
import jax.numpy as jnp
import numpyro
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.diagnostics import print_summary, hdpi
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, init_to_value, Trace_ELBO
import numpyro.optim as optim
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Lịch sử đã quá khắc nghiệt với Ptolemy. Claudius Ptolemy (90-168 sau công nguyên) là một nhà toán học và chiêm tinh gia người Ai Cập, ông được biết đến với mô hình địa tâm trong hệ mặt trời. Ở hiện đại, nếu khoa học muốn chế giễu ai đó, thì họ sẽ ví ông như những kẻ tin vào thuyết địa tâm. Nhưng ông là một thiên tài. Mô hình toán học về chuyển đạo hành tinh ([**HÌNH 4.1**](#f1)) cực kỳ chính xác. Để có được độ chính xác cao, ông dùng thiết bị tên là *epicycle*, tức vòng tròn trên vòng tròn. Có thể có epi-epicycle, vòng tròn trên vòng tròn trên vòng tròn. Nếu số lượng vòng tròn đủ và đúng, mô hình của Ptolemy có thể dự báo chính xác chuyển đạo hành tinh với chính xác cao. Và nên mô hình của ông đã được sử dụng hơn một nghìn năm. Ptolemy và những người như ông đã xây dựng những mô hình này không cần sự hỗ trợ của máy tính. Ai ai cũng sẽ xấu hổ nếu so sánh với Ptolemy.

Vấn đề ở đây dĩ nhiên là mô hình địa tâm là sai, ở nhiều phương diện. Nếu bạn dùng nó để vẽ chuyển đạo của Sao Hoả, bạn sẽ vẽ lệch xa vị trí hành tinh đỏ này một khoảng rất dài. Nhưng mô hình vẫn rất tốt để phát hiện Sao Hoả trên bầu trời đêm. Mặc dù có thể mô hình cần phải được tái chỉnh sau mỗi thập kỷ, tuỳ vào vật thể nào mà bạn muốn định vị. Nhưng mô hình địa tâm vẫn tiếp tục cho ra dự báo chính xác, cho rằng những dự đoán đó nằm trong giới hạn nhỏ của câu hỏi.

![](/assets/images/fig 4-1.png)
<details class="fig"><summary>Hình 4.1: Vũ trụ của Ptolemy, trong đó các chuyển đạo phức tạp của các hành tinh trên bầu trời đêm được giải thích bằng các vòng tròn trong vòng tròn, gọi là <i>epicycle</i>. Mô hình này là sai rõ ràng, nhưng lại cho dự đoán khá tốt.</summary></details>

Phương pháp dùng epicycle có vẻ điên rồ, khi mà bạn biết chính xác cấu trúc của hệ mặt trời. Nhưng trong cổ đại, con người đã dùng nó như biện pháp ước lượng tổng quát hoá. Cho rằng có số lượng vòng tròn đủ trong một không gian đủ, cách làm của Ptolemy giống như *Fourier serries*, một cách phân tách một hàm tuần hoàn (như quỹ đạo) thành một tập hợp hàm sin và hàm cos. Cho nên các hành tinh thực có sắp xếp như thế nào thì mô hình địa tâm vẫn có thể dùng để mô tả quỹ đạo của chúng trên bầu trời.

**HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)** là mô hình địa tâm trong thống kê. "Hồi quy tuyến tính" ở đây là một nhóm golem thống kê đơn giản để tìm trung bình và phương sai của nhiều kết quả đo lường,  bằng cách dùng phép cộng của nhiều kết quả đo lường khác. Giống thuyết địa tâm, hồi quy tuyến tính có thể mô tả rất tốt nhiều hiện tượng tự nhiên đa dạng. Tương tự như thuyết địa tâm, hồi quy tuyến tính là mô hình mô tả tương ứng với nhiều mô hình xử lý khác nhau. Nếu chúng ta đọc cấu trúc của nó theo đúng nghĩa, có thể chúng ta sẽ gây ra sai lầm. Nhưng nếu dùng tốt, những con golem tuyến tính cũng khá hữu dụng.

Chương này giới thiệu hồi quy tuyến tính trong quy trình Bayes. Dưới sự diễn giải bằng xác suất, điều mà cần thiết trong những tác vụ trong Bayes, hồi quy tuyến tính dùng phân phối Gaussian (phân phổi normal - bình thường) để mô tả tính bất định của kết quả đo lường đang quan tâm. Mô hình kiểu này khá đơn giản, linh hoạt và được dùng rất nhiều. Cũng giống tất cả các mô hình thống kê, hồi quy tuyến tính không áp dụng cho mọi trường hợp. Nhưng nó là mô hình cơ bản nhất, vì nếu bạn hiểu cách xây dựng và diễn giải mô hình hồi quy tuyến tính, bạn có thể dễ dàng họcc tiếp những dạng khác của hồi quy mà ít phổ biến hơn. 


## <center>4.1 Tại sao phân phối normal lại normal</center><a name="a1"></a>

Giả sử bạn có một ngàn người xếp hàng ở đường giữa sân bóng. Mỗi người có một đồng xu. Mỗi lần huýt sáo thì họ sẽ lần lượt tung đồng xu. Nếu đồng xu ra mặt ngửa, người đó qua bên trái đường giữa một bước. Nếu đồng xu ra mặt sấp, người đó qua bên phải đường giữa một bước. Mỗi người thực hiện tung đồng xu 16 lần, tuân theo hướng dẫn của từng lần tung và đứng yên sau đó. Giờ chúng ta đo khoảng cách từ mỗi người đến đường giữa, bạn có thể đoán được tỉ lệ nào trong 1000 người đó nằm ở đường giữa không? Còn 5 mét bên trái đường giữa thì sao?

Rất khó để biết cụ thể một người nào sẽ đứng ở đâu, nhưng bạn có thể tự tin và nói rằng tại vị trí nào đó sẽ chiếm tỉ lệ bao nhiêu. Số đo khoảng cách ấy sẽ phân phối gần như theo phân phối normal, hay Gaussian. Điều này đúng ngay khi phân phối gốc là binomial. Nó xảy ra như thế bởi vì có rất nhiều khả năng xảy ra để một trình tự trái phải mà có tổng là zero. Những trình tự mà cách một bước trái hoặc phải thì ít hơn, và cứ thế, với số lượng những trình tự còn lại giảm dần theo khuynh hướng giống đường cong hình chuông của phân phối normal.

### 4.1.1 Normal qua phép cộng

Hãy xem kết quả này, thông qua mô phỏng. Để chứng minh rằng không có gì đặc biệt về cơ chế tung đồng xu, giả định rằng thay vì mỗi bước là khác với những bước khác, là một khoảng cách ngẫu nhiên giữa zero và một mét. Khi đồng xu được tung lên, một khoảng cách giữa zero và một mết phải được thực hiện theo hướng chỉ điểm, và lăph lại quy trình. Để mô phỏng quy trình trên bằng cách tạo cho mỗi người một dãy 16 số gồm -1 và 1 một cách ngẫu nhiên. Đó là những bước độc lập. Sau đó chúng ta tính tổng chúng lại để lấy vị trí sau 16 bước. Sau đó chúng ta lặp lại cho 1000 người. Đây là tác vụ có thể được thực hiện nhiều qua giao diện nhấn chọn, những nó rõ ràng hơn khi dùng code. Đây là đoạn code một dòng để làm toàn bộ những thứ này:

<b>Code 4.1</b>
```python
pos = jnp.sum(dist.Uniform(-1, 1).sample(random.PRNGKey(0), (1000, 16)), -1)
```

<a name="f2"></a>![](/assets/images/fig 4-2.svg)
<details class="fig"><summary>Hình 4.2: Bước đi ngẫu nhiên trên sân bóng sẽ hội tụ lại thành phân phối normal. Càng nhiều bước đi, thì càng giống nhau hơn giữa phân phối vị trí thực tế và phân phối normal lý tưởng, thể hiện ở biểu đồ cuối cùng ở hàng dưới.</summary>
{% highlight python %}fig = plt.figure(figsize=(10,6))
gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0,:])
axes = [fig.add_subplot(gs[1,i]) for i in range(3)]
nums = 200
tosses = dist.Uniform(-1, 1).sample(random.PRNGKey(0), (nums, 16))
cumsum = jnp.cumsum(tosses, axis=1)
cumsum_0 = jnp.concatenate([jnp.zeros(nums).reshape(-1,1), cumsum], axis=1)
ax1.plot(jnp.arange(17), cumsum_0.T, linewidth=0.2, alpha=0.6, color='C0')
ax1.set(xlabel="số bước", ylabel="vị trí", xticks=[0,4,8,16])
for ax,steps in zip(axes,[4,8,16]):
    az.plot_dist(cumsum[:, steps], ax=ax, bw=0.2, rug=True)
    ax.set(title=f"{steps} bước", ylabel="mật độ", xlabel="vị trí")
m, s = jnp.mean(cumsum[:,-1]), jnp.std(cumsum[:,-1])
seq = jnp.linspace(-6,6,100)
axes[2].plot(seq, jnp.exp(dist.Normal(m,s).log_prob(seq)), color="C1", linewidth=2 )
plt.tight_layout(){% endhighlight %}</details>

Bạn có thể vẽ phân phối của vị trí cuối cùng theo một số cách khác nhau, bao gồm `plt.hist` và `az.plot_dist`. Trong [**HÌNH 4.2**](#f2), tôi thể hiện kết quả của bước đi ngẫu nhiên và phân phối của chúng sẽ phát triển như thế nào khi số bước tăng lên. Hàng trên thể hiện 200 người thực hiện bước đi ngẫu nhiên. Các đường gridline tương ứng với phân phối ở hàng dưới, được đo sau bước thứ 4, 8, 16. Mặc dù phân phối ban đầu có vẻ rời rạc hỗn loạn, sau bước thứ 16, nó gần như có hình chuông quen thuộc. Đường cong hình "chuông" quen thuộc của phân phối normal được xuất phát từ sự ngẫu nhiên. Bạn có thể thí nghiệm với số bước nhiều hơn để kiểm tra phân phối khoảng cách có tuân theo phân phối Gaussian hay không. Bạn có thể bình phương khoảng cách giữa 2 bước chân hay biến đổi các số tuỳ ý, kết quả vẫn không thay đổi: Phân phối normal vẫn xuất hiện. Vậy nó đến từ đâu?

Bất kỳ trình xử lý nào có phép cộng tất cả các giá trị ngẫu nhiên đến từ chung một phân phối sẽ hội tụ lại thành phân phối normal. Nhưng không dễ để nắm bắt được tại sao phép cộng lại ra đường cong hình chuông của tổng số.<sup><a name="r65" href="#65">65</a></sup> Bạn có thể suy nghĩ như vậy. Cho dù giá trị trung bình của phân phối nguồn là gì, mỗi lần lấy mẫu từ nó có thể được hiểu là sự dao động của con số trung bình ấy. Khi chúng ta bắt đầu cộng chúng lại, chúng cũng bắt đầu từ bù trừ lẫn nhau. Dao động dương lớn sẽ bù trừ dao động âm lớn. Số lượng các số hạng trong tổng càng nhiều, thì xác suất sẽ nhiều hơn để mỗi dao động bị bù trừ bởi dao động khác, hoặc bởi một tập hợp nhiều dao động nhỏ ngược chiều. Cho nên dần dần giá trị tổng số cuối cùng mà có khả năng xuất hiện cao nhất, là tổng mà mọi dao động đều bị bù trừ, hay giá trị zero (liên quan với trung bình).<sup><a name="r66" href="#66">66</a></sup>

Hình dáng của phân phối nền thì không quan trọng lắm. Nó có thể là phân phối đồng dạng, như ví dụ ở trên, hoặc cũng có thể là bất kỳ phân phối khác.<sup><a name="r67" href="#67">67</a></sup> Tuỳ vào loại phân phối nền mà sự hội tụ có thể diễn ra chậm, nhưng nó vẫn sẽ xảy ra. Thông thường, như ví dụ này, sự hội tụ diễn ra rất nhanh.

### 4.1.2 Normal qua phép nhân

Đây là một cách khác để có được phân phối normal. Giả sử tốc độ phát triển của vi khuẩn bị ảnh hưởng bởi vài loci trong gen, những loci này chứa allel mã hoá sự phát triển. Giả sử tất cả những gen này tương tác với nhau, ví dụ như mỗi gen làm tăng thêm phần trăm sự phát triển. Có nghĩa là hiệu ứng của chúng là phép nhân hơn là phép cộng. Ví dụ, chúng ta có thể lấy mẫu tỉ lệ phát triển ngẫu nhên ở ví dụ này qua dòng code:

<b>Code 4.2</b>
```python
jnp.prod(1 + dist.Uniform(0, 0.1).sample(random.PRNGKey(0), (12,)))
```

Code trên tạo ngẫu nhiên 12 con số từ 1.0 đến 1.1, mỗi số tương ứng với tỉ lệ phát triển. Như 1.0 là không phát triển và 1.1 là tăng 10%. Tích của chúng sẽ phân phối theo normal. Thật vậy ta có thể lấy 1000 con số như vậy và kiểm tra.

<b>Code 4.3</b>
```python
growth = jnp.prod(1 + dist.Uniform(0, 0.1).sample(random.PRNGKey(0), (1000, 12)), -1)
az.plot_dist(growth)
x = jnp.sort(growth)
plt.plot(x, jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x)), "--")
```

Bạn đọc nên chạy code này và sẽ thấy rằng phân phối này gần như là normal. Tôi đã nói phân phối normal xuất hiện từ tổng các dao động ngẫu nhiên, mà thực sự như vậy. Nhưng hiệu ứng của mỗi loci là phép nhân với hiệu ứng của những loci khác, chứ không phải phép cộng. Chuyện gì đã xảy ra?

Chúng ta lần nữa có được sự hội tụ về phân phối normal, là do hiệu ứng nhân tại mỗi loci là quá nhỏ. Phép nhân số nhỏ với nhau có thể được xem như phép cộng. Ví dụ như 2 loci tăng sự phát triển 10%, thì tích là:

$$ 1.1 \times 1.1 =1.21 $$

Ta có thể ước lượng số này bằng phép cộng thay vì phép nhân, và độ lệch chỉ khoảng 0.01:

$$ 1.1 \times 1.1 = (1+0.1)(1+0.1) = 1 +0.2 +0.01 \approx 1.2 $$

Hiệu ứng của loci càng nhỏ, thì ước lượng bằng phép cộng càng tốt. Bằng cách này, hiệu ứng nhỏ nhân với nhau cũng giống như phép cộng, và chúng thường ổn định dần thành phân phối Gaussian. Bạn có thể tự kiểm tra lại:

<b>Code 4.4</b>
```python
big = jnp.prod(1 + dist.Uniform(0, 0.5).sample(random.PRNGKey(0), (1000, 12)), -1)
small = jnp.prod(1 + dist.Uniform(0, 0.01).sample(random.PRNGKey(0), (1000, 12)), -1)
```

Mức độ chênh lệch của, sự phát triển mà có tương tác này, chỉ cần nó đủ nhỏ, sẽ hội tụ thành phân phối Gaussian. Trong trường hợp này, căn nguyên tạo ra phân phối Gaussian đã vươn xa hơn sự tương tác qua phép cộng đơn thuần.

### 4.1.3 Normal qua logarith của phép nhân

Chưa hết, còn nữa. Với mức độ chênh lệch lớn trong phép nhân thì không tạo ra phân phối Gaussian, nhưng nó có xu hướng tạo thành phân phối Gaussian ở thang logarith. Ví dụ:

<b>Code 4.5</b>
```python
log_big = np.log(np.prod(1 + dist.Uniform(0, 0.5).sample(random.PRNGKey(0), (1000, 12)), -1))
```

Nó cũng là phân phối Gaussian. Chúng ta có phân phối Gaussian bởi vì phép cộng các logarith là tương ứng với phép nhân các số ban đầu. Cho nên tương tác phép nhân với độ lệch lớn vẫn có thể tạo ra phân phối Gaussian, nếu chúng ta đo lường kết cục ở thang logarith. Vì thang đo của đo lường là ngẫu nhiên, cho nên không có gì nghi ngờ với sự chuyển đổi này. Dù sao đi nữa, nó rất tự nhiên nếu người ta đo lường âm thanh và động đất, thậm chí thông tin (Chương 7) ở thang logarith.

### 4.1.4 Sử dụng phân phối Gaussian

Chúng ta sẽ sử dụng phân phối Gausssian xuyên suốt chương này để tạo bộ khung cho các giả thuyết, sau đó xây dựng mô hình đo lường là kết hợp của phân phối normal. Lý giải về tại sao dùng phân phối này gồm 2 nhóm: (1) tự nhiên và (2) phương pháp học.

Theo lý giải tự nhiên, thế giới này chứa rất nhiều phân phối Gaussian tương đối. Chúng ta không thể nào trải nghiệm một phân phối Gaussian hoàn hảo. Nhưng những quy luật như nó tồn tại ở nhiều nơi, xuất hiện lặp đi lặp lại ở nhiều thang khác nhau và trong lĩnh vực khác nhau. Sai số đo lường, biến thiên trong sự phát triển, tốc độ của nguyên tử luôn hội tụ thành phân phối Gaussian. Những trình xử lý đó tạo ra điều này bởi vì trong bản chất của chúng là phép cộng các sự dao động. Và phép cộng dao động xảy ra vô hạn sẽ cho ra phân phối của các tổng, phân phối đó chứa tất cả thông tin của trình xử lý bên dưới, ngoài con số trung bình và độ biến thiên ra.

Một trong những hệ quả là các mô hình thống kê dựa trên phân phối Gaussian sẽ không phát hiện tốt những trình xử lý siêu nhỏ. Nó nhắc lại triết lý thiết kế mô hình ở Chương 1. Nhưng nó cũng có nghĩa là những mô hình này có thể làm được nhiều việc hữu ích, ngay khi chúng không xác định được trình xử lý bên dưới. Nếu chúng ta bắt buộc phải biết được quy luật sinh học về phát triển của chiều cao trước khi ta thiết kê mô hình thống kê cho nó, sinh học của loài người đã không phát triển như hiện nay.

Có rất nhiều quy luật trong giới tự nhiên, cho nên đừng nghĩ rằng phân phối Guassian áp dụng được cho mọi thứ. Các chương sau ta sẽ gặp các quy luật hữu dụng và phổ biến như exponential, gamma, Poisson, và chúng đều có từ giới tự nhiên. Phân phối Gaussian là một thành viên trong họ các phân phối tự nhiên nền tảng, còn gọi là **HỌ LUỸ THỪA (EXPONENTIAL FAMILY)**. Tất cả các thành viên của họ này đêu rất quan trọng trong khoa học, bởi vì nó tồn tại ở khắp thế giới của chúng ta.

Nhưng sự xuất hiện trong giới tự nhiên của phân phối Gaussian chỉ là một lý do để thiết kế mô hình dựa trên nó. Theo lý giải phương pháp học, Gaussian đại diện cho một trạng thái thiếu hiểu biết. Nếu những gì chúng ta biết về phân phối của đo lường (những giá trị liên tục thuộc số thực) là trung bình và phương sai, thì phân phối Gaussian là kiên định nhất với giả định của chúng ta.

Nói cách khác là phân phối Gaussian là sự thể hiện tự nhiên nhất về trạng thái thiếu hiểu biết của chúng ta, bởi vì nếu giả định rằng đo lường có phương sai giới hạn, phân phối Gaussian là hình dạng có thể biểu diễn số lượng các cách lớn nhất mà không có giả định mới được đưa vào. Nó ít bất ngờ nhất và cần ít thông tin giả định nhất. Bằng cách này, phân phối Gaussian là kiên định nhất với giả định của golem. Nếu bạn không nghĩ phân phối là Gaussian, thì đó suy ra bạn biết gì đó khác mà cần nói thêm cho golem, và việc này sẽ cải thiện suy luận.

Lý giải phương pháp học này là mở đầu cho **THUYẾT THÔNG TIN (INFORMATION THEORY)** và **TỐI ĐA HOÁ ENTROPY (MAXIMUM ENTROPY)**. Trong các chương sau, các phân phối thường gặp và hữu ích khác sẽ được dùng để thiết kê *mô hình tuyến tính tổng quát (generalized linear model - GLM)*. Khi những phân phối khác này được giới thiệu, bạn sẽ học các ràng buộc để làm chúng trở thành phân phối thích hợp nhất.

Còn bây giờ, ta hãy chấp nhận lý giải tự nhiên và lý giải phương pháp học về việc sử dụng Gaussian để thiết kế mô hình đo lường xung quanh nó. Xuyên suốt quá trình thiết kế này, hãy nhớ rằng sử dụng mô hình không đồng nghĩa ta phải thề thốt gì với nó. Golem là người hầu của chúng ta, không phải ngược lại.

<div class="alert alert-info">
    <p><strong>Những các đuôi lớn.</strong> Phân phối Gaussian thường gặp trong tự nhiên và có một số tính chất đẹp. Nhưng sử dùng nó đi đôi với việc chấp nhận một số nguy cơ. Tận cùng của một phân phối thường được gọi là đuôi của phân phối. Và phân phối Gaussian có hai đuôi rất nhỏ - có rất ít xác suất nằm ở chúng. Ngược lại thì phần lớn mật độ của Gaussian nằm trong giới hạn một độ lệch chuẩn quanh trung bình. Nhiều trình xử lý tự nhiên (và không tự nhiên) có đuôi lớn hơn. Những trình xử lý này có nhiều xác suất hơn để tạo ra giá trị cực. Một ví dụ thực tế và quan trọng là chuỗi thời gian trong kinh tế - sự kiện lên xuống của chứng khoán có thể nhìn giống Gaussian trong thời gian ngắn, nhưng khi thời gian dài ra, những giá trị cực gây sốc sẽ làm cho mô hình Gaussian trở nên ngu ngốc.<sup><a name="r68" href="#68">68</a></sup> Chuỗi  thời gian trong lịch sử cũng tương tự, và suy luận ví dụ như xu hướng chiến tranh thì dễ bị bất ngờ tại những đuôi lớn.<sup><a name="r69" href="#69">69</a></sup> Ta sẽ xem xét những phân phối thay thế cho Gaussian sau.</p>
</div>

<div class="alert alert-dark">
    <p><strong>Phân phối Gaussian.</strong> Bạn không cần phải nhớ phân phối xác suất Gaussian. Máy tính đã biết nó. Nhưng hiểu biết về công thức có thể giúp trả lời những bí mật xoay quanh nó. <i>Hàm mật độ xác suất (Probability density)</i> của giá trị $y$ bất kỳ, trong phân phối Gaussian có trung bình $\mu$ và độ lệch chuẩn $\sigma$, là:</p>
$$ p(y|\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(y-\mu)^2}{2\sigma^2} \right) $$
    <p>Nó thật khủng khiếp. Phần quan trọng là ở $(y-\mu)^2$. Đây là bộ phận cho hình cong parabol nền tảng của phần phối normal. Khi bạn dùng $e$ luỹ thừa parabole, thì bạn có được hình chuông kinh điển. Phần còn lại chỉ là thang (scale) và chuẩn hoá phân phối.</p>
    <p>Gaussian là phần phối liên tục, không phải phân phối rời rạc như chương trước. Hàm xác suất cho kết cục rời rạc, như nhị phân (binomial), gọi là hàm <i>khối lượng xác suất (probability mass)</i> và ký hiệu $\Pr$. Phân phối liên tục như Gaussian thì được gọi là hàm <i>mật độ xác suất (probability density)</i> và ký hiệu $p$ hoặc đơn giản là $f$, tuỳ vào tác giả và nơi làm việc. Vì lý do toán học, mật độ xác suất có thể lớn hơn 1. Hãy thử <code>jnp.exp(dist.Normal(0,0.1).log_prob(0))</code>, ví dụ, nó là cách để tính $p(0 | 0, 0.1)$. Kết quả, ra khoảng 4, là không sai. <i>Mật độ</i> xác suất là tần suất thay đổi trong xác suất tích luỹ. Cho nên ở chỗ nào mà xác suất tích luỹ tăng nhanh, thì mật độ có thể dễ dàng lớn hơn 1. Nhưng nếu ta tính diện tích dưới hàm mật độ, nó không bao giờ lớn hơn 1. Phần diện tích đó cũng có thể gọi là <i>khối lượng xác suất (probability mass)</i>. Chúng ta thường bỏ qua những chi tiết về mật độ/khối lượng khi thực hiện tính toán trên máy tính. Nhưng vẫn có ích nếu có để ý sự phân biệt này. Đôi khi sự khác biệt này có quan trọng.</p>
    <p>Bạn thường gặp phân phối Gaussian không có tham số $\sigma$ mà có tham số khác là $\tau$. Tham số $\tau$ trong tình huống này thường được gọi là <i>độ chính xác</i> và được định nghĩa là $\tau = 1/\sigma^2$. Khi $\sigma$ lớn, $\tau$ nhỏ. Sự thay đổi tham số này cho ta công thức tương đương như sau (chỉ cần thay thế $\sigma=1/\sqrt{\tau}$):</p>
$$ p(y|\mu,\tau) = \sqrt{\frac{\tau}{2\pi}}exp\left( -\frac{1}{2}\tau(y-\mu)^2  \right) $$
    <p>Công thức dưới dạng này thường gặp hơn trong phân tích Bayes, và phần mềm fit mô hình Bayes, như BUGS hoặc JAGS, đôi khi yêu cầu sử dụng $\tau$ thay vì $\sigma$.</p>
</div>

## <center>4.2 Ngôn ngữ mô tả mô hình</center><a name="a2"></a>

Sách này sử dụng một ngôn ngữ tiêu chuẩn để mô tả và mã hoá mô hình thống kê. Bạn sẽ gặp loại ngôn ngữ này trong nhiều sách thống kê và hầu như tất cả tạp chí thống kê, vì nó là tổng quát cho cả thiết kế mô hình Bayes và non-Bayes. Các nhà khoa học ngày càng nhiều hơn trong việc sử dụng chung loại ngôn ngữ này để mô tả các phương pháp thống kê của họ. Cho nên học ngôn ngữ này là một sự đầu tư, cho dù bạn có hướng đi nơi nào tiếp theo.

Cách tiếp cận được tóm tắt như sau. Sẽ có nhiều ví dụ về sau, nhưng trước mắt, thì công thức tổng quát quan trọng hơn.

1. Liệt kê các biến số (variable) mà ta cần làm việc. Một vài biến thì quan sát được, được gọi là *dữ liệu (data)*. Những biến khác thì không quan sát được, như tần suất và trung bình, được gọi là *tham số (parameter)*.
2. Chúng ta định nghĩa những biến số dưới dạng kết hợp của các biến khác hoặc là một phân phối xác suất.
3. Sự kết hợp của các biến số và phân phối xác suất của chúng gọi là *mô hình kết hợp khả tạo (joint generative model)* có thể dùng cho mô phỏng quan sát theo giả thuyết cũng như phân tích data thật.

Các bước tiếp cận này áp dụng cho mô hình ở mọi lĩnh vực, từ thiên văn đến lịch sử nghệ thuật. Khó khăn lớn nhất nằm ở câu hỏi nghiên cứu - biến số nào là quan trọng và giả thuyết giúp liên kết chúng như thế nào - chứ không phải toán học.

Sau khi tất cả sự lựa chọn được quyết định - và đa số chúng sẽ nhanh trở nên quen thuộc đối với bạn - chúng ta tóm tắt mô hình bằng một thứ gì đó giống toán học như sau:

$$ \begin{aligned}
y &\sim \text{Normal} (\mu_i, \sigma) \\
\mu_i &= \beta x_i \\
\beta &\sim \text{Normal} (0,10) \\
\sigma &\sim \text{Exponetial}(1)\\
x_i &\sim \text{Normal}(0,1)
\end{aligned}$$

Nếu bạn thấy nó khó hiểu, tốt. Nó có nghĩa là bạn đang đọc đúng sách, bởi vì sách này sẽ dạy bạn cách đọc và viết những mô tả về mô hình toán học này. Chúng ta không thực hiện tính toán nào cho nó. Thay vào đó, nó cho ta một phương pháp rõ ràng để định nghĩa và giao tiếp với mô hình. Một khi bạn cảm thấy thoải mái với ngữ pháp của chúng, khi bạn bắt đầu đọc những mô tả toán học trong sách khác hoặc tài liệu khoa học, bạn sẽ thấy chúng dễ chịu hơn.

Cách tiếp cận trên chắc chắn không phải phương pháp duy nhất để mô tả thiết kế mô hình thống kê, nhưng nó là một ngôn ngữ phổ biến và hiệu quả. Một khi nhà khoa học học được ngôn ngữ này, thì họ sẽ có thể giao tiếp dễ dàng với những giả định của mô hình. Ta không cần phải nhớ những điều kiện kiểm định khủng khiếp như *đồng phương sai (homoscedasticity)* (phương sai là hằng số), bởi vì chúng ta có thể đọc chúng từ định nghĩa của mô hình. Chúng ta sẽ có thể nhìn thấy cách tự nhiên để thay đổi những giả định, thay vì bị gò bó trong những mô hình bảo thủ như hồi quy hay hồi quy đa biến hay ANOVA hay ANCOVA hay đại loại như thế. Chúng là chung một mô hình, và sự thật đó sẽ trở nên rõ ràng một khi chúng ta biết cách nói chuyện với mô hình, hay cách liên kết một tập hợp các biến số vào trong phân phối xác suất lên một tập biến số khác. Cơ bản, những mô hình này định nghĩa những cách mà giá trị của biến số có thể xuất hiện, dưới giá trị của biến số khác (Chương 2).

### 4.2.1 Quay lại ví dụ mô hình tung quả cầu

Làm việc với ví dụ là một khởi đầu tốt. Nhớ lại vấn đề tỉ lệ nước từ chương trước. Mô hình trong truòng hợp đó sẽ là:

$$ \begin{aligned}
W &\sim \text{Binomial} (N, p) \\
p &\sim \text{Uniform} (0,1) \\
\end{aligned}$$

Trong đó, $W$ là số đếm nước quan sát được, $N$ là số lần tung, và $p$ là tỉ lệ nước trên quả cầu. Mệnh đề trên có thể đọc như sau:

>Số đếm $W$ được phân phối nhị phân (binomial) với cỡ mẫu $N$ và xác suất $p$.  
>Prior cho $p$ được giả định là đồng dạng giữa zero và một.

Một khi biết mô hình theo cách này, chúng ta tự động hiểu được mọi giả định của nó. Chúng ta biết phân phối binomial giả định mỗi mẫu (lần tung) đều độc lập với nhau, cho nên chúng ta cũng biết mô hình giả định mỗi lần tung độc lập với nhau.

Bây giờ, chúng ta sẽ tập trung vào những mô hình đơn giản như trên. Ở những mô hình này, dòng đầu tiên định nghĩa hàm likelihood được dùng trong Bayes' theorem. Những dòng khác định nghĩa prior. Cả hai dòng trong mô hình này **PHÂN PHỐI NGẪU NHIÊN (STOCHASTIC)**, được ký hiệu là $\sim$. Mối quan hệ theo phân phối ngẫu nhiên chẳng qua là một ánh xạ của biến số hoặc tham số đến một phân phối. Nó là *phân phối ngẫu nhiên* bởi vì không một giá trị nào của biến số ở bên trái là được biết cụ thể. Thay vào đó, sự ánh xạ có tính chất xác suất: Vài giá trị thì có tính phù hợp cao hơn, nhưng rất nhiều giá trị khác vẫn có tính phù hợp dưới mô hình bất kỳ. Sau này, chúng ta sẽ gặp mô hình có những định nghĩa khẳng định (deterministic) trong nó.

<div class="alert alert-dark">
    <p><strong>Từ định nghĩa model đến Bayes' theorem.</strong> Để liên hệ với định dạng toán học ở trên với Bayes' theorem, bạn có thể sử dụng định nghĩa mô hình để định nghĩa phân phối posterior:</p>

$$\Pr(p|w,n) = \frac{\text{Binomial}(w|n,p) \text{Uniform}(p|0,1)}{\int \text{Binomial}(w|n,p) \text{Uniform}(p|0,1) dp} $$

<p>Con quái vật ở mẫu số chỉ là xác suất trung bình. Nó chuẩn hoá posterior để có tổng bằng 1. Hành động chính xảy ra ở tử số, nơi mà xác suất posterior của bất kỳ giá trị cụ thể của $p$ được thấy lần nữa tỉ lệ thuận với tích của prior và likelihood. Ở dạng code, nó là phép tính giống như grid approximation mà bạn đang dùng. Dưới dạng dễ dàng nhận ra hơn biểu thức trên:</p>

<b>Code 4.6</b>
{% highlight python %}w,p = 6,9
p_grid = jnp.linspace(0,1,100)
posterior = jnp.exp(dist.Binomial(n, p_grid).log_prob(w) + dist.Uniform(0, 1).log_prob(p_grid))
posterior = posterior/jnp.sum(posterior){% endhighlight %}

Bạn hãy so sánh với phép tính ở các chương trước.</div>

## <center>4.3 Model Gaussian chiều cao</center><a name="a3"></a>

Bây giờ chúng ta sẽ bắt đầu xây dựng mô hình hồi quy tuyến tính (linear regression). Thực ra thì, nó sẽ là "hồi quy" một khi có biến dự đoán (predictor variable) vào. Trước mắt chúng ta sẽ để trống nó và thêm nó lại vào ở phần sau. Hiện tại, chúng ta muốn một biến đo lường trong mô hình có phân phối Gaussian. Sẽ có 2 tham số mô tả hình dạng của phân phối: trung bình $\mu$ và độ lệch chuẩn $\sigma$. Cập nhật Bayes sẽ cho phép ta xem xét mọi cặp có khả năng của $\mu$ và $\sigma$ và cho điểm mỗi kết hợp đó thông qua tính phù hợp tương đối, dưới sự xuất hiện của data. Tính phù hợp tương đối là xác suất posterior của mỗi kết hợp các giá trị $\mu$, $\sigma$.

Nói một cách khác là vầy. Có vô số phân phối Gaussian. Vài cái có trung bình nhỏ. Vài cái khác có trung bình lớn. Vài cái thì rộng, với $\sigma$ lớn. Vài cái khác thì hẹp hơn. Chúng ta muốn cỗ máy Bayes xem xét mọi phân phối khả dĩ, vỗi mỗi phân phối được định nghĩa bằng $\mu$ và $\sigma$, và xếp hạng chúng theo tính phù hợp posterior. Tính phù hợp posterior cho phép một cách đo lường sự thích hợp logic của mỗi phân phối khả dĩ với data và mô hình.

Trong thực hành, chúng ta sẽ dùng các phương pháp ước lượng để thực hiện phân tích. Cho nên chúng ta không thực sự xem xét mọi kết hợp khả dĩ của $\mu$ và $\sigma$. Nhưng chuyện đó sẽ không gây hao phí gì cả. Thay vào đó, điều cần lo lắng là luôn nhớ rằng "sự ước lượng" sẽ là toàn bộ posterior chứ không phải một điểm nào trong đó. Kết quả là, phân phối posterior sẽ là phân phối của các phân phối Gaussian. Đúng vậy, phân phối của các phân phối. Nếu bạn không hiểu, có nghĩa bạn là người trung thực. Cứ tiếp tục, cố gắng, và bạn sẽ hiểu nó sớm thôi.

### 4.3.1 Dữ liệu (Data)

Ta sẽ dùng data `Howell1`là một phần data dân số của Dobe area !Kung San, được tạo ra từ cuộc phỏng vấn thực hiện bởi Nancy Howell ở cuối những năm 1960.<sup><a name="r70" href="#70">70</a></sup> Dành cho những bạn không phải nhà nghiên cứu loài người, thì !Kung San là dân tộc sống bằng săn bắt hái lượm nổi tiếng nhất ở thế kỷ 20, phần lớn bởi vì những nghiên cứu định lượng chi tiết của những người như Howell. Tải dữ liệu vào và gán chúng thành một đối tượng (object):

<b>Code 4.7</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv?raw=true", sep=";")
```

Bây giờ bạn có một *khung dữ liệu (dataframe)* tên là `d`. Tôi sẽ dùng tên `d` nữa và nữa trong sách này để nói đến khung dữ liệu đang thao tác. Tôi chọn tên của nó ngắn như vậy để tiết kiệm số lần gõ máy. Một *khung dữ liệu* là một đối tượng đặc biệt trong package `pandas` của python. Nó là một bảng với các cột có tên, tương ứng với các biến số, và các dòng có đánh số, tương ứng với từng trường hợp riêng lẻ. Trong ví dụ này, các trường hợp là riêng lẻ. Khảo sát cấu trúc của khung dữ liệu ở python:

<b>Code 4.8</b>
```python
d.info()
```
<samp>&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 544 entries, 0 to 543
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   height  544 non-null    float64
 1   weight  544 non-null    float64
 2   age     544 non-null    float64
 3   male    544 non-null    int64  
dtypes: float64(3), int64(1)
memory usage: 17.1 KB</samp>

Sử dụng hàm `head` hoặc `tail` để kiểm tra những dòng dữ liệu đầu hoặc cuối, mặc định số dòng là 5:

```python
d.head()
```
<p><samp><table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>151.765</td>
      <td>47.825606</td>
      <td>63.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>139.700</td>
      <td>36.485807</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>136.525</td>
      <td>31.864838</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>156.845</td>
      <td>53.041915</td>
      <td>41.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>145.415</td>
      <td>41.276872</td>
      <td>51.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table></samp></p>

Chúng ta có thể dùng hàm `describe` hoặc `print_summary` trong `numpyro` để tóm tắt data, và chúng ta cũng sẽ dùng nó để tóm tắt phân phối posterior ở phần sau:

<b>Code 4.9</b>
```python
print_summary(dict(zip(d.columns, d.T.values)), 0.89, False)
```
<samp>            mean       std    median      5.5%     94.5%     n_eff     r_hat
   age     29.34     20.75     27.00      0.00     57.00    186.38      1.03
height    138.26     27.60    148.59     90.81    170.18    218.68      1.06
  male      0.47      0.50      0.00      0.00      1.00    670.75      1.00
weight     35.61     14.72     40.06     11.37     55.71    305.62      1.05</samp>

Dataframe này chứa bốn cột. Mỗi cột có 544 dòng, cho nên có 544 mẫu quan sát riêng lẻ trong data này. Mỗi mẫu quan sát có thu thập chiều cao (centimeter), cân nặng (kilogram), tuổi (năm), và giới tính (0 chỉ điểm cho nữ và 1 chỉ điểm cho nam).

Chúng ta sẽ làm việc với cột `height` lúc này. Cột chứa dữ liệu chiều cao chỉ là một *vector* trong python, tức là một dãy số mà chúng ta đã thao tác trên nó. Bạn có thể lấy vector này bằng tên của nó:

<b>Code 4.10</b>
```python
d["height"]
```

Dấu `[]` thực ra là một hàm dùng để *chỉ điểm (indexing)*, như code trên là *chỉ điểm cột `height` trong dataframe `d`.*

Tất cả những gì chúng ta cần bây giờ là chiều cao của những người lớn trong mẫu. Lý do để lọc ra những mẫu quan sát không phải người lớn bây giờ là chiều cao tương quan rất mạnh với độ tuổi, trước khi trở thành thanh niên. Phần sau chương này, tôi sẽ yêu cầu bạn đối đầu với vấn đề độ tuổi. Nhưng bây giờ, tốt hơn hết là tạm hoãn lại. Bạn có thể chọn lọc dataframe chỉ còn những cá thể lớn hơn hoặc bằng 18:

<b>Code 4.11</b>
```python
d2 = d[d['age']>=18]
```

Bây giờ, chúng ta sẽ làm việc với dataframe `d2`. Nó có 352 dòng (mẫu quan sát) trong đó.

<div class="alert alert-info">
<p><strong>Dataframe và index.</strong> Ký hiệu dấu ngoặc vuông được dùng ở trên gọi là ký hiệu <i>chỉ điểm (index)</i>. Nó rất hữu dụng, nhưng lại quá ngắn gọn và dễ gây nhầm. Dataframe <code>d</code> là một ma trận (matrix), một khung chữ nhật chứa đầy các giá trị, ứng với class <code>DataFrame</code> trong <code>pandas</code>. Để tìm hiểu về <i>indexing</i> của dataframe, bạn đọc có thể tham khảo thêm tại <a href="https://pandas.pydata.org/docs/user_guide/indexing.html">documentation của pandas</a>.</p>
<p>Có vẻ như chúng ta không cần đến dataframe. Nếu chúng ta chỉ cần làm việc với một cột thôi, thì tại sao phải chú ý đến <code>d</code>? Bạn không cần phải dùng dataframe, khi bạn chỉ cho vector thô vào lệnh sẽ dùng trong sách này. Nhưng để giữa các biến liên quan trong cùng một dataframe thì rất tiện lợi. Khi bạn có nhiều hơn một biến số, và bạn muốn mô hình hoá biến này là hàm số của biến kia, việc sử dụng dataframe sẽ có lợi hơn. Bạn không cần phải đợi lâu.</p>
</div>

### 4.3.2 Mô hình

Mục tiêu của chúng ta là mô hình hoá những giá trị này bằng phân phối Gaussian. Trước tiên, ta sẽ vẽ phân phối của chiều cao, `az.plot_dist`. Nó có vẻ Gaussian, giống như dữ liệu chiều cao kinh điển. Có lẽ bởi vì chiều cao là tổng của nhiều yếu tố phát triển nhỏ. Như đã đề cập ở đầu chương, phân phối của tổng thường là hội tụ thành phân phối Gaussian. Dù lý do thế nào, chiều cao của người lớn trong một quần thể thường là gần như hoàn toàn normal.

Tại thời điểm này thì đó là lý do đủ để mô hình dùng phân phối Gaussiancho phân phối xác suất của data. Nhưng hãy cẩn thận nếu bạn chọn Gaussian sau khi thấy biểu đồ phân phối của kết cục giống Gaussian. Chỉ nhìn vào data thô, mà quyết định hướng thiết kế mô hình, thường không phải là ý tưởng tốt. Data có thể là hỗn hợp của nhiều phân phối Gaussian, ví dụ, và trong trường hợp đó bạn không thể nhận biết được tính normal bị ẩn bên dưới chỉ bằng nhìn vào phân phối của kết cục. Hơn nữa, như đã nói ở lúc đầu, phân phối của mẫu quan sát được không nhất thiết phải giống Gaussian để dùng phân phối Gaussian.

Vậy chọn phân phối Gaussian nào? Cố vô số phân phối Gaussian, với vô số trung bình và độ lệch chuẩn. Chúng ta đã chuẩn bị viết xuống mô hình tổng quát và tính toán tính phù hợp của mỗi kết hợp $\mu$ và $\sigma$. Để định nghĩa chiều cao là phân phối normal với trung bình $\mu$ và độ lệch chuẩn $\sigma$, chúng ta viết như sau:

$$ h_i \sim \text{Normal} (\mu, \sigma) $$

Có sách ghi là $h_i \sim \mathcal{N}(\mu,\,\sigma) $, nó là cùng một mô hình với cách ghi khác. Ký hiệu *h* là dánh sách các chiều cao, và *i* nhỏ là vị trí trong data (*index*). Index *i* lấy giá trị số hàng, và trong ví dụ này thì có thể là giá trị từ 1 đến 352 (số lượng các chiều cao trong `d['height']`. Với mô hình này, golem sẽ hiểu mỗi số đo chiều cao được định nghĩa từ chung một phân phối normal với trung bình $\mu$ và $\sigma$. Sau này, *i* nhỏ sẽ xuất hiện ở tay phải trong công thức, và bạn sẽ thấy ý nghĩa của *i* nhỏ. Cho nên đừng mặc kệ nó, mặc dù hiện tại nó có vẻ vô nghĩa.

<div class="alert alert-info">
<p><strong>Phân phối độc lập và như nhau.</strong> Mô hình nhỏ ở trên giả định giá trị $h_i$ là <i>phân phối độc lập và như nhau (independent and identically distributed, i.i.d., iid, hoặc IID)</i>. Bạn có thể thấy mô hình trên được viết như sau:</p>
$$ h_i \stackrel{i.i.d.}{\sim} \text{Normal} (\mu, \sigma) $$
<p>"iid" chỉ rằng mỗi giá trị $h_i$ đều có chung hàm xác suất, độc lập với những giá trị $h$ khác và sử dụng cùng tham số. Ta có cảm giác điều này có vẻ không đúng. Ví dụ, chiều cao trong chung gia đình thường sẽ tương quan bởi vì có chung allel đến từ dòng họ chung.</p>
<p>Giả định của i.i.d. không có gì đáng ngại cả, chỉ cần nhớ rằng xác suất nằm trong golem, không phải thế giới lớn. Giả định i.i.d. là cách golem đại diện cho tính bất định của nó. Đây là một giả định về mặt <i>phương pháp học</i>. Nó không phải giả định thực thể về thế giới, về mặt <i>tự nhiên</i>. E.T.Jaynes (1922-1998) gọi đây là <i>sự lừa đảo do ánh xạ tâm trí (mind projection fallacy)</i>, lỗi lầm do hiểu sai giữa lý do về phương pháp học và lý do tự nhiên.<sup><a name="r71" href="#71">71</a></sup> Trọng điểm ở đây là không phải phương pháp học tốt hơn tự nhiên, nhưng với sự hiểu biết về mối tương quan này, thì "i.i.d." là phân phối tốt nhất.<sup><a name="r72" href="#72">72</a></sup> Bạn sẽ gặp lại vấn đề này ở Chương 10. Hơn nữa, có một hệ quả toán học được biết đến là <i>de Finetti's theorem</i>, nó nói rằng các giá trị có thể <strong>THAY ĐỔI ĐƯỢC (EXCHANGABLE)</strong> thì có thể được ước lượng từ hỗn hợp nhiều phân phối iid. Nói dễ hiểu thì giá trị thay đổi được thì có thể tái sắp xếp được. Tác động thực tế của "i.i.d." thì không thể hiểu được theo nghĩa đen. Có một vài loại tương quan có thể thay đổi không nhiều về hình dạng của phân phối, chúng ảnh hưởng đến trình tự xuất hiện của giá trị. Ví dụ, cặp sinh đôi có tương quan cao về chiều cao. Nhưng phân phối chung của chiều cao vẫn là normal. Markov chain Monte Carlo (Chương 9) lợi dụng điều này, bằng cách sử dụng những mẫu theo trình tự có tương quan cao để ước lượng mọi phân phối chúng ta cần.</p></div>

Để hoàn thành mô hình, ta cần priors. Tham số cần ước lượng là cả $\mu$ và $\sigma$, cho nên ta cần prior $\Pr(\mu,\sigma)$, hay xác suất kết hợp của tất cả các tham số. Trong đa số trường hợp, prior thường được định nghĩa cụ thể độc lập cho từng tham số, hay tương đương với  giả định $\Pr(\mu,\sigma) = \Pr(\mu)\Pr(\sigma)$. Sau đó chúng ta có thể viết:

$$ \begin{aligned}
h_i &\sim \text{Normal} (\mu, \sigma) && \quad [\text{likelihood}]\\
\mu &\sim \text{Normal} (178,20) && \quad [\mu \; \text{prior}]\\
\sigma &\sim \text{Uniform} (0, 50) && \quad [\sigma \; \text{prior}]
\end{aligned} $$

Những thông tin ở bên phải không phải là thành phần của mô hình, mà chỉ là ghi chú để giúp bạn theo dõi mục đích của mỗi dòng trong mô hình. Prior của $\mu$ là một prior Gaussian khá rộng, trung bình ở 178 cm, với 95% xác suất giữa 178 $\pm$ 40 cm.

Tại sao 178 cm? Tác giả  của sách nàycao 178 cm. Khoảng cách từ 138 cm đến 218 cm đảm bảo một khoảng lớn về thông tin chiều cao ở dân số loài người. Vậy thông tin kiến thức chuyên môn đã có trong prior. Mọi người đều biết gì đó về chiều cao con người và có thể cài đặt prior đúng logic cho đại lượng này. Nhưng trong nhiều bài toán hồi quy, sử dụng thông tin prior sẽ khó khăn hơn vì tham số nhiều khi không có ý nghĩa thực thể.

Cho dù prior là gì, thì việc vẽ prior ra là một ý tưởng tốt, để bạn có được một tầm nhìn về giả định mà họ đưa vào mô hình. Trong trường hợp này:

<b>code 4.12</b>
```python
x = np.linspace(100, 250, 101)
plt.plot(x, np.exp(dist.Normal(178, 20).log_prob(x)));
```

Thực thi đoạn mã trên, bạn sẽ thấy golem này giả định là chiều cao trung bình (không phải từng chiều cao riêng lẻ) là hầu như chắc chắn trong khoảng giữa 140 cm và 220 cm. Cho nên prior này chứa một vài thông tin, nhưng nó không nhiều. Prior $\sigma$ là prior phẳng thực sự, là phân phối đồng dạng, nhiệm vụ của nó chỉ để ràng buộc $\sigma$ luôn luôn dương trong khoảng giá trị từ 0 đến 50 cm. Bạn có thể xem nó qua:

<b>code 4.13</b>
```python
x = jnp.linspace(-10, 60, 101)
plt.plot(x, jnp.exp(dist.Uniform(0, 50, validate_args=True).log_prob(x)))
```

Độ lệch chuẩn như $\sigma$ phải là số dương, cho nên giới hạn nó ở zero là hợp lý. Làm sao để chọn giới hạn trên? Độ lệch chuẩn 50cm sẽ cho rằng 95% chiều cao cá nhân sẽ nằm trong 100cm trên dưới chiều cao trung bình. Nó là khoảng rất lớn.

Câu chuyện nãy giờ là tốt. Nhưng tốt hơn khi ta nhìn thấy những prior này sẽ cho phân phối chiều cao cá nhân như thế nào. Mô phỏng **DỰ ĐOÁN PRIOR (PRIOR PREDICTIVE)** là một phần quan trọng trong thiết kế mô hình. Khi bạn chọn xong prior cho $h$, $\mu$ và $\sigma$, chúng suy ra một phân phối xác suất kết hợp (joint distribution) cho chiều cao riêng lẻ. Bằng cách mô phỏng từ phân phối này, ta sẽ thấy được lựa chọn của bạn suy ra chiều cao quan sát được sẽ như thế nào. Điều này sẽ giúp bạn chẩn đoán những lựa chọn xấu. Rất nhiều lựa chọn thuận tiện là lựa chọn không tốt, và chúng ta có thể phát hiện điều đó thông qua mô phỏng dự đoán prior.

Được rồi, vậy ta làm như thế nào? Bạn có thể mô phỏng nhanh các giá trị chiều cao bằng cách lấy mẫu từ prior, như bạn đã lấy mẫu từ posterior trong Chương 3. Nhớ không, mỗi posterior cũng là một prior tiềm năng cho những lần phân tích sau, cho nên bạn có thể xử lý prior cũng như posterior.

<b>code 4.14</b>
```python
sample_mu = dist.Normal(178, 20).sample(random.PRNGKey(0), (int(1e4),))
sample_sigma = dist.Uniform(0, 50).sample(random.PRNGKey(1), (int(1e4),))
prior_h = dist.Normal(sample_mu, sample_sigma).sample(random.PRNGKey(2))
az.plot_kde(prior_h, bw=1);
```

<a name="f3"></a>![](/assets/images/fig 4-3.svg)
<details class="fig"><summary>Hình 4.3: Mô phỏng dự đoán prior cho mô hình chiều cao. Hàng trên: phân phối prior cho $\mu$ và $\sigma$. Hàng dưới bên trái: Mô phỏng dự đoán prior cho chiều cao, sử dụng prior ở hàng trên. Giá trị ở 3 độ lệch chuẩn được thể hiện ở trục hoành. Hàng dưới bên phải: Mô phỏng dự đoán prior sử dụng $\mu \sim \text{Normal} (178,100)$.</summary>{% highlight python %}sample_mu = dist.Normal(178, 20).sample(random.PRNGKey(0), (int(1e4),))
sample_sigma = dist.Uniform(0, 50).sample(random.PRNGKey(1), (int(1e4),))
prior_h = dist.Normal(sample_mu, sample_sigma).sample(random.PRNGKey(2))
sample_mu2 = dist.Normal(178, 100).sample(random.PRNGKey(0), (int(1e4),))
prior_h2 = dist.Normal(sample_mu2, sample_sigma).sample(random.PRNGKey(2))
fig, axs = plt.subplots(2,2,figsize=(12,10))
axs[0,0].plot(jnp.linspace(100,250,100), jnp.exp(dist.Normal(178,20).log_prob(jnp.linspace(100,250,100))))
axs[0,0].set(xticks=[100,178,250], ylabel="mật độ", xlabel="mu", title="mu ~ Norrmal(178, 20)")
axs[0,1].plot(jnp.linspace(-10,60,100), jnp.exp(dist.Uniform(0,50,True).log_prob(jnp.linspace(-10,60,100))))
axs[0,1].set(xticks=[0,50], xlabel="sigma", ylabel="mật độ", title="sigma ~ Uniform(0, 50)")
az.plot_kde(prior_h, bw=1, ax=axs[1,0])
axs[1,0].set(xticks=[0,73,178,283], xlabel="chiều cao", ylabel="mật độ", title="h ~ Normal(mu, sigma)")
az.plot_kde(prior_h2, bw=3, ax=axs[1,1])
axs[1,1].set(xlabel="chiều cao", ylabel="mật độ", xticks=[-128,0,178,484],title="h ~ Normal(mu, sigma)\nmu~Normal(178,100)")
axs[1,1].axvline(0,0,1, linestyle="dashed", linewidth=3)
axs[1,1].axvline(272,0,1, linewidth=2)
plt.tight_layout(){% endhighlight %}</details>

Mật độ này, cũng như mật độ cụ thể cho $\mu$ và $\sigma$, được thể hiện ở [**HÌNH 4.3**](#f3). Nó hiện lên một hình chuông với hai đuôi lớn. Đây là phân phối mong đợi cho chiều cao, trung bình hoá trên prior. Chú ý rằng phân phối xác suất prior của chiều cao không phải tự nó Gaussian. Điều này không sao. Phân phối bạn thấy không giống như mong đợi theo kiến thức thực tế, nhưng là phân phối các tính phù hợp tương đối của chiều cao khác nhau, trước khi thấy data.

Mô phỏng dự đoán prior có thể hữu ích trong việc lựa chọn các prior hợp lý, cho nên rất khó để đánh giá prior ảnh hưởng thế nào đến biến quan sát được. Ví dụ xem xét prior phẳng hơn và chứa ít thông tin hơn cho $\mu$, như $\mu \sim \text{Normal} (178, 100)$. Prior với độ lệch chuẩn lớn rất thường gặp trong mô hình Bayes, nhưng chúng đa số là không hợp lý. Hãy thử mô phỏng lần nữa để xem những giá trị chiều cao suy ra từ nó:

<b>code 4.15</b>
```python
sample_mu = dist.Normal(178, 100).sample(random.PRNGKey(0), (int(1e4),))
prior_h = dist.Normal(sample_mu, sample_sigma).sample(random.PRNGKey(2))
az.plot_kde(prior_h, bw=1);
```

Kết quả được thể hiện trên biểu đồ dưới phải của [**HÌNH 4.3**](#f3). Bây giờ mô hình, trước khi thấy được data, cho rằng có khoảng 4% người, nằm ở bên trái của đường nét đứt, có chiều cao là số âm. Nó cũng mong đợi người khổng lồ. Trong lịch sử loài người, người cao nhất là Robert Pershing Wadlow (1918-1940) cao 272 cm. Trong mô phỏng dự đoán prior của chúng ta, có đến 18% người cao hơn chiều cao này (bên phải đường nét liền).

Có ảnh hưởng không? Trong trường hợp này, ta có quá nhiều data nên những prior ngu ngốc này là không gây hại. Nhưng không phải lúc nào cũng vậy. Có rất nhiều vấn đề suy luận mà chỉ data là không đủ, cho dù nhiều cỡ nào. Bayes cho phép chúng ta tiếp tục trong những trường hợp này. Nhưng chỉ khi chúng ta dùng kiến thức khoa học của mình để tạo ra những prior hợp lý. Sử dụng kiến thức khoa học để tạo prior không phải là gian lận. Quan trọng ở đây là prior của bạn không phải dựa trên giá trị trong data, mà là những gì bạn biết về data trước khi thấy nó.

<div class="alert alert-info">
<p><strong>Tạm biệt epsilon.</strong> Nhiều bạn đọc chắc từng thấy ký hiệu thay thế cho mô hình tuyến tính Gaussian như sau:</p>
$$ \begin{aligned}
h_i &= \mu + \epsilon_i \\
\epsilon_i &\sim \text{Normal}(0, \sigma)\\
\end{aligned} $$  
<p>Định nghĩa này tương đương với dạng $h_i \sim \text{Normal}(\mu,\sigma)$, với $\epsilon$ thay thế cho mật độ Gaussian. Nhưng dạng này là một dạng không tốt. Lý do là nó thường không tổng quát hoá được cho các loại mô hình khác. Có nghĩa là nó không thể nào diễn đạt được mô hình không dùng Gaussian bằng thủ thuật như $\epsilon$. Tốt hơn hết là học một hệ thống có thể tổng quát hoá được.</p></div>

<div class="alert alert-dark">
<p><strong>Định nghĩa mô hình theo Bayes' theorem lần nữa.</strong> 
Có thể sẽ tốt nếu nhìn cách định nghĩa mô hình ở trên cho ra phân phối posterior như thế nào. Mô hình chiều cao, với prior cho $\mu$ và $\sigma$, sẽ cho phân phối posterior:</p>

$$ \Pr(\mu,\sigma|h) = \frac{\Pi_i\text{Normal}(h_i|\mu,\sigma)\text{Normal}(\mu|178,20)\text{Uniform}(\sigma|0,50)}{\int\int\Pi_i\text{Normal}(h_i|\mu,\sigma)\text{Normal}(\mu|178,20)\text{Uniform}(\sigma|0,50)d\mu d\sigma} $$

<p>Nó thật khủng khiếp, nhưng nó cũng chính là con quái vật ở trên. Có hai thứ mới làm cho nó trở nên phức tạp. Thứ nhất là có nhiều hơn một quan sát của $h$, cho nên để có được likelihood kết hợp của toàn bộ data, ta phải tính xác suất của mỗi $h$ và nhân chúng lại với nhau. Tích ở bên tay phải sẽ giải quyết chuyện đó. Vấn đề phức tạp thứ hai là có đến hai prior, $\mu$ và $\sigma$. Những chúng xếp lên nhau. Trong grid approximation dưới đây, bạn sẽ thấy định nghĩa này được diễn đạt bằng code. Mọi thứ sẽ được tính trên thang logarith, để phép nhân thành phép cộng. Nhưng dù sao nó cũng là thi hành theo Bayes' theorem.</p></div>

### 4.3.3 Grid approximation cho phân phối posterior

Bởi vì đây là mô hình Gaussian đầu tiên trong sách này, và thực ra nó cũng là mô hình đầu tiên có nhiều hơn một tham số, nó đáng để tìm ra posterior bằng các phép tính bạo lực. Mặc dù tôi không khuyên dùng cách tiếp cận ở nơi khác, bởi vì nó rất vất vả và nặng máy. Thực vậy, nó rất không thực dụng đến nổi nhiều lúc bất khả thi. Nhưng cũng giống như mọi khi, nó đáng để biết mục tiêu chính xác nhìn như thế nào, trước khi chấp nhận các hình dạng thông qua ước lượng. Trong phần sau của chương này, bạn sẽ dùng quadratic approximation để ước tính phân phối posterior, và nó là cách tiếp cận bạn sẽ dùng xuyên suốt nhiều chương đầu. Khi bạn có được mẫu mà bạn tạo ra trong phần này, bạn có thể so sánh kết quả đó với quadratic approximation ở phần sau.

Không may là, để thực hiện phép tính này cần nhiều mánh kỹ thuật và một ít, nếu có, trực quan về các khái niệm. Cho nên tôi sẽ trình bày code này mà không có giải thích. Bạn có thể thực thi nó và đi tiếp, nhưng sau đó quay lại và đi theo endnote để có sự giải thích về thuật toán.<sup><a name="r73" href="#73">73</a></sup> Bây giờ, đây là nội tạng của con golem:

<b>code 4.16</b>
```python
mu_list = np.linspace(start=150, stop=160, num=100)
sigma_list = np.linspace(start=7, stop=9, num=100)
mesh = np.meshgrid(mu_list, sigma_list)
post = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
post["LL"] = vmap(lambda mu, sigma: np.sum(dist.Normal(mu, sigma).log_prob(
    d2.height.values)))(post["mu"], post["sigma"])
logprob_mu = dist.Normal(178, 20).log_prob(post["mu"])
logprob_sigma = dist.Uniform(0, 50).log_prob(post["sigma"])
post["prob"] = post["LL"] + logprob_mu + logprob_sigma
post["prob"] = np.exp(post["prob"] - np.max(post["prob"]))
```

Bạn có thể kiểm tra phân phối posterior, bây giờ đang nằm trong `post["prob"]`, bằng nhiều lệnh vẽ biểu đồ khác nhau. Bạn có thể làm một biểu đồ contour (plot) đơn giản như sau:

<b>code 4.17</b>
```python
plt.contour(post["mu"].reshape(100, 100), post["sigma"].reshape(100, 100),
            post["prob"].reshape(100, 100));
```

Hoặc bạn có thể vẽ biểu đồ heat map đơn giản:

<b>code 4.18</b>
```python
plt.imshow(post["prob"].reshape(100, 100),
           origin="lower", extent=(150, 160, 7, 9), aspect="auto");
```

### 4.3.4 Lấy mẫu từ posterior

Để tìm hiểu posterior này chi tiết hơn, lần nữa tôi sẽ sử dụng cách tiếp cận linh hoạt để lấy mẫu các giá trị tham số từ nó. Nó hoạt động giống như trong Chương 3, khi bạn lấy mẫu các giá trị $p$ từ phân phối posterior trong ví dụ tung quả cầu. Cái mới ở đây là bởi vì có đến hai tham số, và ta cần lấy mẫu các kết hợp của chúng, đầu tiên ta chọn ngẫu nhiên một lượng nhất định các số hàng trong `post` tỉ lệ với các giá trị trong `post["prob"]`. Sau đó chúng ta lấy mẫu các giá trị tham số bằng những con số hàng ngẫu nhiên đó. Đoạn code này làm việc đó:

<b>code 4.19</b>
```python
prob = post["prob"] / np.sum(post["prob"])
sample_rows = dist.Categorical(probs=prob).sample(random.PRNGKey(0), (int(1e4),))
sample_mu = post["mu"][sample_rows]
sample_sigma = post["sigma"][sample_rows]
```

Ta sẽ có được 10,000 mẫu có chọn lại, từ posterior của data chiều cao. Hãy nhìn vào những mẫu này:

<b>code 4.20</b>
```python
plt.scatter(sample_mu, sample_sigma, s=64, alpha=0.1, edgecolor="none");
```

<a name="f4"></a>![](/assets/images/fig 4-4.svg)
<details class="fig"><summary>Hình 4.4 Mẫu rút ra từ phân phối posterior từ data chiều cao. Mật độ các điểm lớn nhất ở trung tâm, phản ánh tính phù hợp cao nhất của cặp $\mu$ và $\sigma$. Có nhiều cách để những giá giá trị tham số nào tạo ra data, đặt điều kiện trên mô hình.</summary>{% highlight python %}plt.scatter(sample_mu, sample_sigma, s=64, alpha=0.1, edgecolor="none")
plt.xlabel("sample_mu")
plt.ylabel("sample_sigma"){% endhighlight %}</details>

Chú ý đối số `alpha`. Nó giúp cho màu sắc nhạt hơn, để [**HÌNH 4.4**](#f4) thể hiện mật độ dễ dàng hơn, khi các mẫu bị chồng lắp. Tuỳ chính biểu đồ theo ý bạn bằng thay đổi `s` (size - kích thước các điểm), `marker` (hình dạng của điểm), `alpha` (độ xuyên thấu),...

Bây giờ bạn đã có những mẫu tham số này, bạn có thể mô tả phân phối của độ tin cậy của mỗi cặp $\mu$ và $\sigma$ thông qua việc tóm tắt các mẫu. Hãy nghĩ chúng như data và mô tả chúng, như bạn đã làm ở Chương 3. Ví dụ, để biết hình dáng của mật độ posterior biên (marginal) của $\mu$ và $\sigma$, bạn chỉ cần:

<b>code 4.21</b>
```python
az.plot_dist(sample_mu)
az.plot_dist(sample_sigma)
```

Cụm từ "biên (marginal)" ở đây nghĩa là "trung bình hoá trên những tham số khác". Mật độ này sẽ rất giống với phân phối bình thường. Và nó khá điển hình. Khi cỡ mẫu tăng lên, mật độ posterior tiến dần đến phân phối normal. Nếu bạn nhìn kỹ, bạn sẽ nhận ra rằng mật độ $\sigma$ có đuôi bên phải dài hơn. Tôi sẽ phân tích xu hướng này sau, để cho bạn thấy tình trạng này rất thường gặp ở tham số độ lệch chuẩn.

Để tóm tắt chiều rộng của các mật độ này bằng khoảng tin cậy posterior:

<b>code 4.22</b>
```python
hpdi(sample_mu, 0.89)
hpdi(sample_sigma, 0.89)
```

Bởi vì những mẫu này chỉ là vector các con số, bạn có thể tính bất kỳ chỉ số thống kê nào từ chúng, mà bạn có thể thực hiện được trên data thông thường: `mean`, `median`, hoặc `quantile`.

<div class="alert alert-dark">
<p><strong>Cỡ mẫu và tính bình thường của posterior của $\sigma$.</strong> Trước khi di chuyển đến phương pháp quadratic approximation như đường tắt để thực hiện suy luận, nó đáng để lặp lại thí nghiệm data chiều cao trên, nhưng bây giờ chỉ với một phần của data gốc. Lý do để làm việc này là để trình diễn, về mặt nguyên tắc, posterior không phải lúc nào cũng có hình dạng Gaussian. Sẽ không có vấn đề gì với số trung bình, $\mu$. Với likelihood Gaussian và prior Gaussian trên $\mu$, phân phối posterior luôn luôn cũng là Gaussian, bất kể cỡ mẫu. Nhưng độ lệch chuẩn $\sigma$ sẽ gây rắc rối. Cho nên nếu bạn quan tâm đến $\sigma$ - người ta thường không - bạn phải cần cẩn trọng khi tận dụng quadratic approximation.</p>
<p>Lý do sâu thẳm cho việc posterior của $\sigma$ thường có đuôi bên phải dài thì khá phức tạp. Nhưng có một cách hiểu vấn đề khá hữu ích là phương sai phải là số dương. Kết quả là, sẽ có nhiều tính bất định về việc phương sai lớn cỡ nào (hay độ lệch chuẩn) hơn là nó nhỏ cỡ nào. Ví dụ, nếu phương sai được ước lượng gần bằng zero, thì bạn biết là nó không thể nhỏ hơn nữa. Nhưng nó có thể lớn hơn rất nhiều.</p>
<p>Hãy phân tích nhanh chỉ với 20 chiều cao từ data chiều cao để làm sáng tỏ điều này. Để lấy mẫu ngẫu nhiên 20 chiều cao từ danh sách gốc:</p>
    <b>code 4.23</b>
{% highlight python %}d3 = d2.height.sample(n=20){% endhighlight %}
<p>Bây giờ bạn lặp lại tất cả code ở phần trên, chỉnh lại để tập trung vào 20 chiều cao ở <code>d3</code> thay vì data gốc. Tôi đã nén tất cả code ở trên vào đây.</p>
<b>code 4.24</b>
{% highlight python %}mu_list = jnp.linspace(start=150, stop=170, num=200)
sigma_list = jnp.linspace(start=4, stop=20, num=200)
mesh = jnp.meshgrid(mu_list, sigma_list)
post2 = {"mu": mesh[0].reshape(-1), "sigma": mesh[1].reshape(-1)}
post2["LL"] = vmap(
    lambda mu, sigma: jnp.sum(dist.Normal(mu, sigma).log_prob(d3.values))
)(post2["mu"], post2["sigma"])
logprob_mu = dist.Normal(178, 20).log_prob(post2["mu"])
logprob_sigma = dist.Uniform(0, 50).log_prob(post2["sigma"])
post2["prob"] = post2["LL"] + logprob_mu + logprob_sigma
post2["prob"] = jnp.exp(post2["prob"] - jnp.max(post2["prob"]))
prob = post2["prob"] / jnp.sum(post2["prob"])
sample2_rows = dist.Categorical(probs=prob).sample(random.PRNGKey(0), (int(1e4),))
sample2_mu = post2["mu"][sample2_rows]
sample2_sigma = post2["sigma"][sample2_rows]
plt.scatter(sample2_mu, sample2_sigma, s=64, alpha=0.1, edgecolor="none"){% endhighlight %}
<p>Sau khi thực thi đoạn code trên, bạn sẽ thấy một biểu đồ điểm (scatter plot) khác của những mẫu lấy ra từ mật độ posterior, nhưng lần này bạn sẽ nhận thấy có một đuôi dài đặc biệt ở phía trên đám mây các điểm. Bạn cũng nên kiểm tra mật độ posterior biên cho $\sigma$, trung bình hoá trên $\mu$, tạo ra bằng:</p>
{% highlight python %}az.plot_kde(sample2_sigma)
x = jnp.sort(sample2_sigma)
plt.plot(x, jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x)), "--"){% endhighlight %}
<p>Đoạn code này có kèm theo ước lượng normal với cùng trung bình và phương sai. Bây giờ bạn có thể thấy được posterior của $\sigma$ không phải là Gaussian, mà có một đuôi dài hướng đến các giá trị lớn hơn.</p></div>

### 4.3.5 Tìm phân phối posterior bằng quadratic approximation

Bây giờ chúng ta sẽ bỏ lại grid approximation và tiếp tục với một trong những động cơ tuyệt vời nhất trong thống kế áp dụng, **QUADRATIC APPROXIMATION**. Mục tiêu của quadratic approximation, hãy nhớ lại, là một phương pháp tiện lợi để cho ra suy luận về hình dạng của posterior. Đỉnh của posterior sẽ nằm ở ước lượng **MAXIMUM A POSTERIORI (MAP)**, và chúng ta có thể có một bức tranh hữu ích về hình dạng của posterior bằng cách sử dụng ước lượng từ quadratic approximation của phân phối posterior tại đỉnh của nó.

Để thực hiện quadratic approximation, chúng ta sẽ dùng `SVI` và guide là `AutoLaplaceApproximation`, nằm trong package `numpyro`. Đầu tiên ta dựng mô hình là một hàm số (function) trong python. Nó sẽ bao gồm các hàm số nguyên thuỷ của `numpyro`, như `sample`. Các hàm `sample` sẽ là các prior, nhưng nếu nó có đối số `obs` thì hàm đó là likelihood. Còn guide thực chất cũng là phân phối với các tham số, và `SVI` sẽ thực hiện tối ưu hoá tham số trong guide dựa vào data. Guide `AutoLaplaceApproximation` mặc định mọi posterior là normal. Cách hoạt động của `SVI` (Stochastic Variational Inference) là nhận lần lượt mô hình được xây dựng như trên, guide `AutoLaplaceApproximation`, thuật toán tối ưu hoá, hàm mất mát, và data. `SVI` này sẽ sử dụng những định nghĩa này và tiến hành lấy mẫu theo xác suất posterior của các kết hợp giá trị tham số. Sau đó nó có thể leo phân phối posterior và tìm đỉnh MAP của nó. Cuối cùng, nó ước lượng đường cong bậc hai tại MAP để tạo ra ước lượng cho phân phối posterior. Luôn nhớ rằng: quy trình này cũng giống như những gì quy trình non-Bayes khác thực hiện, chỉ là không có prior.

Để bắt đầu, ta lặp lại đoạn code tải data và chọn lọc ra những người lớn.

<b>code 4.26</b>
```python
d = pd.read_csv("../data/Howell1.csv", sep=";")
d2 = d[d["age"] >= 18]
```

Bây giờ chúng ta chuẩn bị định nghĩa mô hình, sử dụng cú pháp trong `numpyro`. Định nghĩa mô hình trong trường hợp này cũng giống như trước, nhưng chúng ta sẽ lặp lại nó với từng dòng code `numpyro` tương ứng nằm ở bên phải.

$$ \begin{aligned}
h_i &\sim \text{Normal}(\mu, \sigma)  &&\quad {\Tiny numpyro.sample("height", dist.Normal(mu, sigma), obs=height)}\\
\mu &\sim \text{Normal}(178, 20)  &&\quad {\Tiny mu = numpyro.sample("mu", dist.Normal(178, 20))}\\
\sigma &\sim \text{Uniform}(0, 50)  &&\quad {\Tiny sigma = numpyro.sample("sigma", dist.Uniform(0, 50))}\\
\end{aligned}$$

Giờ hãy đưa những đoạn code trên vào trong một hàm (function) python, nhận đối số là data.

<b>code 4.27</b>
```python
def flist(height):
    mu = numpyro.sample("mu", dist.Normal(178, 20))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
```

Fit mô hình vào data ở dataframe `d2` với:

<b>code 4.28</b>
```python
m4_1 = AutoLaplaceApproximation(flist)
svi = SVI(flist, m4_1, optim.Adam(1), Trace_ELBO(), height=d2.height.values)
p4_1, losses = svi.run(random.PRNGKey(0), 2000)
```

Sau khi thực thi đoạn code này thì bạn có các thông tin posterior lưu ở guide `m4_1`. Để lấy mẫu từ nó và tóm tắt posterior, ta làm như sau:

<b>code 4.29</b>
```python
samples = m4_1.sample_posterior(random.PRNGKey(1), p4_1, (1000,))
print_summary(samples, 0.89, False)
```
<samp>           mean       std    median      5.5%     94.5%     n_eff     r_hat
   mu    154.60      0.40    154.60    154.00    155.28    995.06      1.00
sigma      7.76      0.30      7.76      7.33      8.26   1007.15      1.00</samp>

Các con số này là những ước lượng Gaussian cho *phân phối biên* của mỗi tham số. Có nghĩa là tính phù hợp của mỗi giá trị của $\mu$, sau khi trung bình hoá trên mọi phù hợp của $\sigma$, là một phân phối Gaussian với trung bình là 154.6 và độ lệch chuẩn 0.4.

Phân vị 5.5% và 94.5% là ranh giới khoảng percentile, tương ứng với khoảng tin cậy 89% (CI 89%). Tại sao lại 89%? Đơn giản là tôi thích. Nó hiển thị một khoảng khá rộng, cho nên nó cho ta thấy khoảng xác suất cao của các giá trị tham số. Nếu bạn thích khoảng khác, như khoảng 95% tiện lợi, bạn có thể thay đổi con số `0.89` trong `print_summary`. Nhưng tôi không khuyến khích khoảng 95%, bởi vì nhiều người đọc sẽ khó khăn hơn để không xem chúng như trong các phép kiểm định. 89 là số nguyên tố, cho nên nếu ai đó hỏi bạn về lý giải để sử dụng nó, bạn có thể nhìn thẳng vào họ và tuyên bố, "Bởi vì nó là số nguyên tố". Nó là một lý giải không tệ hơn lý giải thuận tiện của con số 95%.

Tôi khuyến khích bạn so sánh khoảng 89% này với khoảng tin cậy từ ước lượng grid approximation trước. Bạn sẽ thấy chúng là y hệt nhau. Khi posterior gần giống Gaussian, thì đây là những gì bạn nên mong đợi.

Prior ở trên rất yếu, bởi vì nó gần như phẳng và bởi vì có nhiều data. Nếu tôi thêm nhiều thông tin cho prior hơn, bạn sẽ thấy sự ảnh hưởng. Chỉ cần thay độ lệch chuẩn của $\mu$ thành 0.1, thì nó là một prior hẹp.

<div class="alert alert-dark">
<p><strong>Giá trị khởi đầu cho guide.</strong> <code>SVI</code> ước lượng posterior bằng cách leo nó như leo núi. Để làm việc này, nó cần phải bắt đầu từ một vị trí nào đó, tại vài sự kết hợp các giá trị tham số. Nếu bạn không nói nơi bắt đầu, guide trong <code>SVI</code> sẽ chọn ngẫu nhiên các giá trị từ prior. Nhưng ta có quyền lựa chọn giá trị bắt đầu với bất kỳ tham số nào trong mô hình. Trong ví dụ ở phần trước, thì tham số là $\mu$ và $\sigma$. Đây là những giá trị bắt đầu tốt trong trường hợp này:</p>
<b>code 4.30</b>
{% highlight python %}start = {"mu": d2.height.mean(), "sigma": d2.height.std()}
m4_1 = AutoLaplaceApproximation(flist, init_loc_fn=init_to_value(values=start))
svi = SVI(flist, m4_1, optim.Adam(0.1), Trace_ELBO(), height=d2.height.values)
p4_1, losses = svi.run(random.PRNGKey(0), 2000){% endhighlight %}
<p>Những giá trị này là ước chừng tốt cho vị trí thô của giá trị MAP.</p></div>

Những prior mà chúng ta sử dụng thì rất yếu, bởi vì chúng là gần như phẳng và bởi vì có quá nhiều data. Cho nên tôi sẽ sử dụng prior nhiều thông tin hơn cho $\mu$, để bạn có thể thấy được hiệu ứng đó. Tất cả những gì tôi sẽ làm là thay đổi độ lệch chuẩn của prior thành 0.1, để nó thành prior rất hẹp.

<b>code 4.31</b>
```python
def model(height):
    mu = numpyro.sample("mu", dist.Normal(178, 0.1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


m4_2 = AutoLaplaceApproximation(model)
svi = SVI(model, m4_2, optim.Adam(1), Trace_ELBO(), height=d2.height.values)
p4_2, losses = svi.run(random.PRNGKey(0), 2000)
samples = m4_2.sample_posterior(random.PRNGKey(1), p4_2, (1000,))
print_summary(samples, 0.89, False)
```
<samp>           mean       std    median      5.5%     94.5%     n_eff     r_hat
   mu    177.86      0.10    177.86    177.72    178.03    995.05      1.00
sigma     24.57      0.94     24.60     23.01     25.96   1012.88      1.00
</samp>

Chú ý rằng ước lượng của $\mu$ di chuyển rất ít từ prior. Prior tập trung rất nhiều quanh 178. Cho nên không có gì bất ngờ. Nhưng cũng để ý rằng ước lượng của $\sigma$ đã thay đổi rất nhiều, mặc dù chúng ta không đổi prior của nó gì cả. Một khi golem khẳng định trung bình là ở 178 - như prior đã nhấn mạnh - thì con golem phải ước lượng $\sigma$ với điều kiện được đặt trên sự thật đó. Kết quả là một posterior khác cho $\sigma$, mặc dù tất cả những gì chúng ta thay đổi là thông tin prior của một tham số khác.

### 4.3.6 Lấy mẫu từ guide AutoLaplaceApproximation

Phần trên giới thiệu phương pháp ước lượng quadratic approximation cho posterior, thông qua công cụ `SVI`. Mặc dù `numpyro` cung cấp công cụ lấy mẫu khá tiện dụng là `guide.sample_posterior`. Nhưng bạn cũng nên hiểu thuật toán trong nó. Việc lấy mẫu cũng khá đơn giản, nhưng nó không rõ ràng, và nó yêu cầu bạn nhận ra rằng quadratic approximation cho phân phối posterior với nhiều hơn một chiều tham số - $\mu$ và $\sigma$ là hai chiều tham số riêng - chỉ là phân phối Gaussian đa chiều.

Hệ quả là, khi `numpyro` tìm quadratic approximation, nó tính không những độ lệch chuẩn của toàn bộ parameter, mà còn hiệp phương sai (covariance) giữa các cặp tham số. Cũng giống như trung bình và độ lệch chuẩn (hay bình phương của nó, phương sai) là điều kiện đủ cho mô tả phân phối Gaussian đơn chiều, một danh sách các trung bình và một ma trận phương sai và hiệp phương sai là điều kiện đủ để mô tả phân phối Gaussian đa chiều. Để tính ma trận phương sai và hiệp phương sai, cho mô hình `m4_1`, sử dụng:

<b>code 4.32</b>
```python
samples = m4_1.sample_posterior(random.PRNGKey(1), p4_1, (1000,))
vcov = jnp.cov(jnp.stack(list(samples.values()), axis=0))
```
<samp>                mu        sigma 
   mu 0.1697395865 0.0002180593
sigma 0.0002180593 0.0849057933</samp>

Đây là ma trận **PHƯƠNG SAI - HIỆP PHƯƠNG SAI (VARIANCE-COVARIANCE)**. Nó là keo dính đa chiều của quadratic approximation, bởi vì nó nói cho chúng ta biết các tham số liên quan với nhau như thế nào trong phân phối posterior. Ma trận phương sai - hiệp phương sai có thể chia ra làm 2 thành phần:
1. Một vector của các variance cho mỗi tham số
2. Một ma trận tương quan nói cho chúng ta biết khi thay đổi bất kỳ tham số sẽ dẫn đến sự thay đổi có tương quan của những tham số khác như thế nào.

Sự phân chia này thường sẽ dễ hiểu hơn. Cho nên chúng ta sẽ tìm nó:

<b>code 4.33</b>
```python
print(np.diagonal(vcov))
print(vcov / np.sqrt(np.outer(np.diagonal(vcov), np.diagonal(vcov))))
```
<samp>        mu      sigma
0.16973959 0.08490579
               mu       sigma
   mu 1.000000000 0.001816412
sigma 0.001816412 1.000000000</samp>

Vector hai giá trị ở trên là danh sách các variance. Nếu bạn lấy căn bậc hai của vector này, bạn sẽ có được độ lệch chuẩn giống như từ kết quả của `print_summary`. Ma trận 2x2 dưới là ma trận tương quan. Mỗi giá trị là một hệ số tương quan, được giới hạn giữa $-1$ và $+1$, cho mỗi cặp tham số. Giá trị 1 nghĩa là tham số tương quan với chính nó. Những giá trị khác thì thường gần zero hơn, và chúng rất gần zero trong ví dụ này. Điều này nghĩa là khi biết $\mu$, ta sẽ không biết gì thêm về $\sigma$ và tương tự khi biết $\sigma$. ta sẽ không biết gì thêm về $\mu$. Điều này là kinh điển trong loại mô hình Gaussian này. Nhưng thông thường thì nó hiếm gặp, và bạn sẽ thấy ở những chương sau.

Được rồi, vậy làm sao để lấy mẫu từ posterior đa chiều này? Bây giờ thay vì lấy 1 mẫu từ một phân phối Gaussian đơn giản, chúng ta lấy mẫu các vector giá trị từ phân phối Gaussian đa chiều. `numpyro` cung cấp một hàm đơn giản để thực hiện chuyện đó:

<b>code 4.34</b>
```python
post = m4_1.sample_posterior(random.PRNGKey(1), p4_1, (int(1e4),))
pd.DataFrame(post)[:6]
```
<p><samp><table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mu</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>154.242050</td>
      <td>7.559958</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154.483170</td>
      <td>7.306348</td>
    </tr>
    <tr>
      <th>2</th>
      <td>154.976929</td>
      <td>7.279916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154.210190</td>
      <td>7.811492</td>
    </tr>
    <tr>
      <th>4</th>
      <td>155.489166</td>
      <td>7.904957</td>
    </tr>
    <tr>
      <th>5</th>
      <td>154.824768</td>
      <td>7.978177</td>
    </tr>
  </tbody>
</table></samp></p>

Kết quả là một DataFrame với 2 cột với 10,000 ($1e4$) dòng, một cột cho $\mu$ và cột kia cho $\sigma$. Mỗi giá trị là một mẫu từ posterior, cho nên trung bình và độ lệch chuẩn cho mỗi cột sẽ rất gần với kết quả MAP phần trước. Bạn có thể kiểm tra lại bằng tóm tắt mẫu này.

<b>code 4.35</b>
```python
print_summary(post, 0.89, False)
```
<samp>           mean       std    median      5.5%     94.5%     n_eff     r_hat
   mu    154.61      0.41    154.61    153.94    155.25   9926.98      1.00
sigma      7.75      0.29      7.74      7.28      8.22   9502.46      1.00</samp>

Bạn có thể dùng `az.plot_dist` để kiểm tra mức độ giống nhau của `post` so với kết quả từ grid approximation trong [**HÌNH 4.4**](#f4). Những mẫu này cũng bảo tồn hiệp phương sai giữa $\mu$ và $\sigma$. Điều này bây giờ không quan trọng lắm, bởi vì $\mu$ và $\sigma$ không tương quan với nhau gì cả trong mô hình này. Nhưng một khi bạn thêm một biến dự đoán vào mô hình, hiệp phương sai sẽ có rất nhiều ý nghĩa.

<div class="alert alert-dark">
<p><strong>Bên dưới việc lấy mẫu đa biến.</strong> Hàm số <code>print_summary</code> là tiện lợi. Nó chạy mô phỏng đơn giản giống như bạn đã thực hiện ở cuối Chương 3. Công việc này được hoàn thành bởi phiên bản đa chiều của <code>dist.Normal</code>, <code>dist.MultiVariateNormal</code>. Hàm số <code>dist.Normal.sample</code> sẽ mô phỏng các giá trị Gaussian ngẫu nhiên, trong khi <code>dist.MultiVariateNormal.sample</code> mô phỏng các vector giá trị Gaussuan ngẫu nhiên. Để thực hiện điều đó:</p>
<b>code 4.36</b>
{% highlight python %}samples_flat = jnp.stack(list(post.values()))
mu, sigma = jnp.mean(samples_flat), jnp.cov(samples_flat)
post = dist.MultivariateNormal(mu, sigma).sample(random.PRNGKey(0), (int(1e4),)){% endhighlight %}
<p>Bạn thường sẽ không dùng <code>dist.MultiVariateNormal</code> trực tiếp, nhưng đôi khi bạn muốn mô phỏng kết cục Gaussian đa chiều. Trong trường hợp đó, bạn sẽ cần tiếp cận <code>dist.MultiVariateNormal</code> trực tiếp. Và dĩ nhiên điều đó là tốt khi bạn biết một ít về cách cỗ máy vận hành. Sau này, chúng ta sẽ làm việc với phân phối posterior và không thể ước lượng đúng bằng cách này.</p></div>

## <center>4.4 Dự đoán tuyến tính</center><a name="a4"></a>

Những gì chúng ta đã tạo ra làm một mô hình Gaussian về chiều cao trong quần thể người lớn. Nhưng nó không có cảm giác "hồi quy" nào cả. Cụ thể, chúng ta quan tâm đến mô hình hoá kết cục sẽ liên quan như thế nào đến biến số khác, **BIẾN DỰ ĐOÁN (PREDICTOR VARIABLE)**. Nếu biến dự đoán có liên quan thống kê đến biến kết cục, thì chúng ta có thể dùng nó để dự đoán kết cục. Khi biến dự đoán được đưa vào mô hình theo một cách cụ thể, chúng ta sẽ có được hồi quy tuyến tính.


Vậy bây giờ hãy nhìn vào chiều cao ở những người dân Kalahari (biến kết cục) sẽ liên quan với cân nặng (biến dự đoán) như thế nào. Đây không phải câu hỏi khoa học thú vị nhất, tôi biết. Nhưng nó là mối liên quan đơn giản nhất để bắt đầu, và nếu bạn thấy nhàm chán, thì đó bởi vì bạn không có lý thuyết về phát triển và lịch sử sự sống trong kiến thức của bạn. Nếu bạn có, thì nó sẽ rất thú vị. Chúng ta sẽ sau đó thử thêm vào những điều thú vị khác, khi bạn xem xét lại trong ví dụ này từ góc nhìn nhân quả. Bây giờ, tôi chỉ yêu cầu bạn tập trung vào cơ chế ước lượng mối liên quan giữa hai biến số.

Ta đi tiếp bằng cách vẽ biểu đồ điểm giữa chiều cao và cân nặng:

<b>code 4.37</b>
```python
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"))
```

Biểu đồ này không được thể hiện ở đây. Bạn nên tự làm nó. Một khi có biểu đồ, bạn sẽ thấy rõ ràng có một mối quan hệ ở đây: Biết trọng lượng của một người thì có thể giúp bạn dự đoán được chiều cao của người đó.

Để sự quan sát mơ hồ này trở thành mô hình định lượng chính xác hơn mà có thể liên quan những giá trị của `weight` đến những giá trị phù hợp của $height$, chúng ta cần thêm nhiều công nghệ. Làm sao chúng ta lấy mô hình Gaussian từ phần trước và kết hợp các biến dự đoán?

<div class="alert alert-info">
<p><strong>"Hồi quy" là gì?</strong> Nhiều loại mô hình khác nhau được gọi là "hồi quy". Từ này xuất phát từ việc sử dụng nhiều hơn một biến dự đoán để mô hình hoá phân phối nhiều hơn một biến dự đoán. Nguồn gốc của từ này, tuy nhiên, xuất phát từ quan sát của nhà nhân chủng học Francis Galton (1822-1911) về những người con của người cao và người thấp thường gần giống với trung bình của dân số, hay <i>hồi quy về trung bình</i>.<sup><a name="r74" href="#74">74</a></sup></p>
<p>Lý do nhân quả cho hồi quy về trung bình rất đa dạng. Trong trường hợp chiều cao, lý giải nhân quả là thành phần chính của nền tảng khoa học gene quần thể. Nhưng hiện tượng này từ khía cạnh thống kê xuất phát từ việc đo lường các nhân được gán vào trong phân phối chung, dẫn đến <i>hiện tượng thu gọn (shrinkage)</i> khi mỗi giá trị đo lường liên quan đến giá trị khác. Trong bối cảnh dữ liệu chiều cao của Galton, việc cố gắng dự đoán chiều cao của con trai chỉ dựa vào chiều cao của cha là ngu ngốc. Tốt hơn hết là sử dụng quần thể của cha. Điều này dẫn đến dự đoán cho mỗi người con sẽ giống với mỗi người cha nhưng "thu gọn" lại thành trung bình. Dự đoán này sẽ tốt hơn. Hiện tượng hồi quy/thu gọn giống vậy có thể áp dụng vào những khái niệm cao hơn và tạo thành nền tảng của thiết kế mô hình đa tầng (Chương 13).</p>
</div>

### 4.4.1 Chiến lược thiết kế mô hình tuyến tính

Mục tiêu là tạo ra tham số cho trung bình của phân phối Gaussian, $\mu$, thành hàm tuyến tính của các biến dự doán và những tham số mới khác mà chúng ta sáng tạo ra. Chiến lược này thường được gọi đơn giản là **MÔ HÌNH TUYẾN TÍNH (LINEAR MODEL)**. Chiến lược mô hình tuyến tính sẽ hướng dẫn golem giả định rằng các biến dự đoán có một hằng số và quan hệ cộng với trung bình của kết cục. Sau đó golem tính phân phối posterior của mối quan hệ hằng định này.

Có nghĩa là, hãy nhớ lại, cỗ máy sẽ xem xét mọi kết hợp khả dĩ của tham số. Với mô hình tuyến tính, vài tham số sẽ là độ mạnh của quan hệ giữa trung bình outcome, $\mu$, và giá trị của biến số khác. Với mỗi kết hợp của giá trị, cỗ máy sẽ tính xác suất posterior, tức là các giá trị đo lường tính phù hợp tương đối, giả định với mô hình và data. Vì vậy, phân phối posterior sẽ xếp hạng vô số kết hợp khả dĩ của các giá trị tham số thông qua tính phù hợp logic của chúng. Kết quả là, phân phối posterior cung cấp tính phù hợp tương đối của các độ mạnh của quan hệ khả dĩ khác nhau, với các giả định được bạn lập trình vào mô hình. Chúng ta là yêu cầu con golem: "Xem xét tất cả các đường thẳng nối một biến số đến biến số khác. Hãy xếp hạng tất cả những đường thẳng này bằng tính phù hợp, sau khi thấy data". Con golem sẽ trả lời bằng phân phối posterior.

Đây là cách hoạt động của nó, trong trường hợp đơn giản nhất với chỉ một biến dự đoán. Chúng ta sẽ đợi đến chương sau để đối đầu với nhiều hơn một biến dự đoán. Hẫy nhớ lại mô hình Gaussian cơ bản:

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu, \sigma) \quad &&[\text{likelihood}]\\
\mu &\sim \text{Normal}(178,20) \quad &&[\mu \; \text{prior}]\\
\sigma &\sim \text{Uniform}(0,50) \quad &&[\sigma\; \text{prior}]\\
\end{aligned}$$

Bây giờ chúng ta làm thế nào để đưa trọng lượng vào mô hình Gaussian về chiều cao? Giả sử x là tên cho cột các giá trị trọng lượng, `d2['weight']`. Giả sử trung bình của các giá trị x là $\bar{x}$. Bây giờ chúng ta có một biến dự đoán $x$, nó là một danh sách các giá trị đo lường có cùng chiều dài với $h$. Để đưa `weight` vào mô hình, chúng ta định nghĩa trung bình $\mu$ là hàm số của các giá trị trong $x$. Nó sẽ giống như như vậy, với lời giải thích sau đó:

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu_i, \sigma) \quad &&[\text{likelihood}]\\
\mu_i &=\alpha + \beta (x_i - \bar{x}) \quad &&[\text{linear model}]\\
\alpha &\sim \text{Normal}(178, 20) \quad &&[\alpha \;\text{prior}]\\
\beta &\sim \text{Normal} (0, 10) \quad &&[\beta \;\text{prior}]\\
\sigma &\sim \text{Uniform} (0, 50) \quad &&[\sigma \;\text{prior}]\\
\end{aligned}$$

Lần nữa, tôi đã dán nhãn cho từng dòng ở bên phải, các loại định nghĩa mà nó mã hoá. Chúng ta sẽ thảo luận từng thứ một.

#### 4.4.1.1 Xác suất của data.

Hãy bắt đầu bằng xác suất của chiều cao quan sát được, là dòng đầu tiên của mô hình. Nó là gần như y hệt với lúc trước, ngoại trừ bây giờ có thêm vị trí (index) $i$ ở $\mu$ cũng như $h$. Bạn có thể đọc $h_i$ là "mỗi $h$"" và $\mu_i$ là "mỗi $\mu$". Trung bình $\mu$ bây giờ phụ thuộc vào giá trị cụ thể ở mỗi hàng $i$. Cho nên ký hiệu $i$ ở $\mu_i$ chỉ điểm *trung bình phụ thuộc vào từng hàng*.

#### 4.4.1.2 Mô hình tuyến tính

Trung bình $\mu$ không còn là tham số cần ước lượng. Mà là, như bạn thấy ở dòng thứ hai của mô hình, $\mu_i$ được xây dựng từ những tham số khác, $\alpha$ và $\beta$, và biến quan sát $x$. Dòng này không phải quan hệ ngẫu nhiên phân phối (stochastic) - không có dấu $\sim$, mà là dấu = trong nó - bởi vì định nghĩa của $\mu_i$ là mang tính quyết định (deterministic). Nói một cách khác, một khi chúng ta biết $\alpha$ và $\beta$ và $x_i$, chúng ta chắc chắn biết $\mu_i$.

Giá trị $x_i$ chỉ là giá trị trọng lượng ở hàng $i$. Nó chỉ điểm tới cùng một cá thể có chiều cao $h_i$, nằm ở cùng hàng. Tham số $\alpha$ và $\beta$ thì bí ẩn hơn. Chúng từ đâu? Chúng ta tự tạo ra chúng. Tham số $\mu$ và $\sigma$ là điều kiện cần và đủ để mô tả một phân phối Gaussian. Còn $\alpha$ và $\beta$ thực ra là những thiết bị chúng ta sáng tạo ra để kiểm soát $\mu$, cho phép nó thay đổi có hệ thống xuyên suốt mọi trường hợp trong data.

Bạn sẽ tạo được nhiều dạng tham số hơn khi kỹ năng bạn tốt hơn. Một cách khác để hiểu những tham số tự tạo này là nghĩ chúng như mục tiêu để cỗ máy học. Mỗi tham số là một thứ gì đó phải được mô tả ở phân phối posterior. Cho nên khi bạn muốn biết gì đó về data, bạn hỏi golem bằng cách tạo ra một tham số cho nó. Từ từ bạn sẽ hiểu rõ hơn khi bạn tiếp tục học. Đây là cách nó hoạt động trong bối cảnh này. Dòng thứ hai của định nghĩa mô hình chỉ là:

$$ \mu_i = \alpha + \beta(x_i - \bar{x})$$

Nó nói cho golem hồi quy rằng bạn đang có 2 câu hỏi về trung bình của kết cục.
1. Chiều cao mong đợi khi $x_i = \bar{x}$? Tham số $\alpha$ sẽ trả lời câu hỏi này, bởi vì khi $x_i = \bar{x}$, $\mu_i = \alpha$. Với lý do này, $\alpha$ thường được gọi là *intercept*. Nhưng ta không nên nghĩ nó là gì đó trừu tượng, mà nên nghĩ nó là một ý nghĩa liên quan với data.
2. Chiều cao thay đổi như thế nào khi $x_i$ tăng 1 đơn vị? Tham số $\beta$ sẽ trả lời câu hỏi này. Nó thường được gọi là *slope*, lần nữa là một khái niệm trừu tượng. Tốt hơn hết là nên nghĩ nó như tần suất thay đổi mong đợi.

Kết hợp lại thì hai tham số này sẽ yêu cầu golem tìm một đường thẳng để nối $x$ đến $h$, đường thẳng này sẽ đi qua $\alpha$ khi $x_i = \bar{x}$ và có *slope* là $\beta$. Đây là một tác vụ mà golem làm rất tốt. Tuy nhiên nó phụ thuộc vào bạn, hãy chắc chắn bạn đã cho câu hỏi tốt.

<div class="alert alert-info">
<p><strong>Không có gì đặc biệt hay tự nhiên về mô hình tuyến tính.</strong> Chú ý rằng không có gì đặc biệt về mô hình tuyến tính, thực sự vậy. Bạn có thể chọn mối quan hệ khác giữa $\alpha$ và $\beta$ và $\mu$. Ví dụ, định nghĩa dưới đây là hoàn toàn khả thi với $\mu_i$:</p>
$$ \mu_i = \alpha \exp(-\beta x_i)$$
<p>Định nghĩa này không phải hồi quy tuyến tính, nhưng nó cũng định nghĩa một mô hình hồi quy. Quan hệ tuyến tính mà chúng ta dùng là để cho thuận tiện, nhưng không có gì bắt buộc ta phải dùng nó. Nó rất phổ biến ở vài lĩnh vực, như kinh tế hay dân số, dưới dạng hàm số của $\mu$ đến từ giả thuyết, hơn là từ tính địa tâm của mô hình tuyến tính. Mô hình được dựng trên các giả thuyết ủng hộ có thể có hiệu năng tốt hơn nhiều so với mô hình tuyến khi dùng trên cùng một hiện tượng.<sup><a name="r75" href="#75">75</a></sup> Chúng ta sẽ quay lại vấn đề ở phần sau cuốn sách.</p></div>

<div class="alert alert-dark">
<p><strong>Đơn vị và các mô hình hồi quy.</strong> Người đọc đã có kiến thức về vật lý sẽ biết làm thế nào để đặt đơn vị thông qua biểu thức đại loại như vậy. Để thuận lợi, đây là mô hình lúc nãy (không có prior để gọn lại), bây giờ được thêm vào đơn vị ở sau các ký hiệu.</p>

$$ \begin{aligned}
h_i\text{cm} &\sim \text{Normal}(\mu_i\text{cm}, \sigma\text{cm})\\
\mu_i\text{cm}&=\alpha\text{cm}+ \beta \frac{\text{cm}}{\text{kg}}(x_i \text{kg} - \bar{x}\text{kg})\\
\end{aligned}$$

<p>Bạn sẽ thấy rõ $\beta$ phải có đơn vị là cm/kg để cho trung bình $\mu_i$ có đơn vị là cm. Một trong những sự thật là gán đơn vị cho chúng sẽ là rõ hơn tham số như $\beta$ là một loại tần suất - centimet trên kilogram. Cũng có một truyền thống là <i>phân tích vô chiều (dimensionless analysis</i> cho rằng xây dựng các biến số sao cho chúng là những tỉ số không cần đơn vị. Trong bối cảnh này, ví dụ, chúng ta có thể cho chiều cao chia cho một chiều cao tham khảo, loại bỏ đơn vị của nó. Đơn vị đo lường là một cấu trúc ngẫu nhiên của con người, nên đôi khi phân tích không cần đơn vị thì tự nhiên và tổng quát hơn.</p></div>

#### 4.4.1.3 Prior

Các dòng còn lại mô tả phân phối prior của biến không quan sát được. Những biến số này thường được gọi là tham số (parameter), và phân phối của chúng là prior. Ở đây có ba tham số: $\alpha, \beta, \sigma$. Bạn đã gặp prior $\alpha$ và $\sigma$, mặc dù $\alpha$ ở  phần trước được gọi là $\mu$.

Prior $\beta$ có lẽ cần nhiều giải thích hơn. Tại sao lại có prior là Gaussian với trung bình là zero? Prior này đặt xác suất dưới zero và lớn hơn zero là bằng nhau, và khi $\beta = 0$, trọng lượng sẽ không liên quan với chiều cao. Để dánh giá prior này suy ra gì, chúng ta phải mô phỏng phân phối dự đoán prior.

Mục tiêu là mô phỏng chiều cao từ mô hình, chỉ sử dụng prior. Trước tiên, ta xem xét khoảng giá trị trọng lượng dùng để mô phỏng. Có thể dùng khoảng giá trị của trọng lượng quan sát được. Sau đó chúng ta  cần mô phỏng các đường thẳng, những đường thẳng được suy ra từ prior của $\alpha$ và $\beta$. Bạn sẽ thực hiện như sau, việc cài đặt `seed` sẽ giúp bạn tái tạo lại kết quả một cách chính xác:

<b>code 4.38</b>
```python
with numpyro.handlers.seed(rng=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 20), sample_shape=(N,))
    b = numpyro.sample("b", dist.Normal(0, 10), sample_shape=(N,))
```

Bây giờ bạn có 100 cặp $\alpha$ và $\beta$. Để vẽ các đường thẳng:

<b>code 4.39</b>
```python
plt.subplot(xlim=(d2.weight.min(), d2.weight.max()), ylim=(-100, 400),
            xlabel="weight", ylabel="height")
plt.axhline(y=0, c="k", ls="--")
plt.axhline(y=272, c="k", ls="-", lw=0.5)
plt.title("b ~ Normal(0, 10)")
xbar = d2.weight.mean()
x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
for i in range(N):
    plt.plot(x, a[i] + b[i] * (x - xbar), "k", alpha=0.2)
```

<a name="f5"></a>![](/assets/images/fig 4-5.svg)
<details class="fig"><summary>Hình 4.5 Mô phỏng dự đoán prior cho mô hình chiều cao và trọng lượng. Bên trái: Mô phỏng dùng prior $\beta \sim \text{Normal}(0,10)$. Bên phải: Prior hợp lý hơn $\beta \sim \text{LogNormal}(0,1)$.</summary>
{% highlight python %}with numpyro.handlers.seed(rng_seed=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 20), sample_shape=(N,))
    b1 = numpyro.sample("b1", dist.Normal(0, 10), sample_shape=(N,))
    b2 = numpyro.sample("b2", dist.LogNormal(0, 1), sample_shape=(N,))
fig, axs = plt.subplots(1,2, figsize=(12,5))
for idx, b in enumerate([b1,b2]):
    axs[idx].set(xlim=(d2.weight.min(), d2.weight.max()), ylim=(-100, 400),
                xlabel="trọng lượng", ylabel="chiều cao")
    axs[idx].axhline(y=0, color="0.5",ls="--")
    axs[idx].axhline(y=272, color="0.5", ls="-", lw=3)
    xbar = d2.weight.mean()
    x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
    for i in range(N):
        axs[idx].plot(x, a[i] + b[i] * (x - xbar),'C0', alpha=0.2)
axs[0].set_title('b ~ Normal(0,10)')
axs[1].set_title('b ~ LogNormal(0,1)')
axs[1].annotate("Người cao nhất thế giới (272cm)", (32,280))
axs[1].annotate("Phôi", (32,10))
plt.tight_layout(){% endhighlight %}</details>

Kết quả được thể hiện ở [**HÌNH 4.5**](#f5). Để tham khảo, tôi đã thêm một đường nét đứt ở zero - không ai thấp hơn zero - và đường "Wadlow" ở 272 cm, người cao nhất thế giới. Các đường thẳng trông không giống như loài người chút nào. Nó nói rằng mối quan hệ giữa trọng lượng và chiều cao có thể là âm tính hoặc dương tính quá mức. Trước khi kịp nhìn thấy data, thì đây là một mô hình kém. Chúng ta có thể nào làm tốt hơn?

Chúng ta có thể làm tốt hơn ngay bây giờ. Chúng ta biết rằng chiều cao trung bình tăng theo cân nặng trung bình, ít ra dưới một mức nào đó. Hãy thử giới hạn nó bằng giá trị dương. Cách đơn giản nhất là định nghĩa prior bằng Log-Normal. Nếu bạn không quen với logarith, không sao cả. Sẽ có thông tin chi tiết hơn phần thông tin cuối.

Định nghĩa $\beta$ theo Log-Normal(0,1) có nghĩa là tuyên bố logarith của $\beta$ là phân phối Normal(0,1). Cụ thể:

$$ \beta \sim \text{Log-Normal}(0,1)$$

Trong `numpyro` có sẵn hàm `dist.LogNormal` để làm việc với phân phối log-normal. Bạn có thể mô phỏng mối quan hệ này để xem nó có ý nghĩa như thế nào với $\beta$:

<b>code 4.40</b>
```python
b = dist.LogNormal(0, 1).sample(random.PRNGKey(0), (int(1e4),))
az.plot_kde(b)
```

Nếu logarith của $\beta$ là normal, thì bản thân $\beta$ sẽ luôn là số dương. Lý do là $\exp(x)$ hay $e^x$ luôn lớn hơn zero với mọi số thực $x$. Nó là lý do prior Log-Normal rất thường gặp. Chúng là một cách dễ dàng để ràng buộc mối quan hệ dương. Vậy nó giúp ta cái gì? Hãy mô phỏng dự đoán prior lần nữa, với prior là Log-Normal:

<b>code 4.41</b>
```python
with numpyro.handlers.seed(rng_seed=2971):
    N = 100  # 100 lines
    a = numpyro.sample("a", dist.Normal(178, 28).expand([N]))
    b = numpyro.sample("b", dist.LogNormal(0, 1).expand([N]))
```

Biểu đồ này được thể hiện ở bên phải của [**HÌNH 4.5**](#f5). Nó hợp lý hơn rất nhiều. Vẫn có một mối quan hệ hiếm và bất khả dĩ. Nhưng đa số các đường thẳng ở prior kết hợp của $\alpha$ và $\beta$ là nằm trong khoảng bình thường của con người.

Chúng ta đang tuỳ biến prior, mặc dù bạn sẽ thấy rằng ở phần sau vì có quá nhiều data trong ví dụ này nên việc lựa chọn prior không còn ý nghĩa nữa. Có 2 lý do để chúng ta tuỳ biến prior. Một, có rất nhiều phân tích mà không kích cỡ data nào làm cho prior không liên quan. Trong trường hợp đó, quy trình non-Bayes cũng không làm được gì hơn. Chúng cũng phụ thuộc vào đặc trưng cấu trúc của mô hình. Chú ý đến việc chọn prior lúc ấy là rất cần thiết. Thứ hai, suy nghĩ về prior giúp ta phát triển mô hình tốt hơn, thậm chí có thể dần dần vượt qua thuyết địa tâm.

<div class="alert alert-info">
<p><strong>Prior đúng là cái nào?</strong> Nhiều người hỏi rằng prior đúng cho một phân tích đưa ra là cái nào. Câu hỏi này đôi khi suy ra rằng với một data nào đó, thì có một prior đúng duy nhất, nếu không nghiên cứu sẽ không hợp lệ. Điều này là sai. Không có nhiều prior đúng duy nhất hơn likelihood đúng duy nhất. Mô hình thống kê là cỗ máy cho suy luận. Nhiều cỗ máy sẽ hoạt động được, nhưng vài cái trong chúng là tốt hơn những cỗ máy khác. Prior có thể sai, nhưng điều đó cũng giống như việc dùng một cây búa không thích hợp để làm một chiếc bàn.</p>
<p>Để chọn prior, có nhiều hướng dẫn đơn giản giúp bạn bắt đầu. Prior mã hoá tình trạng thông tin trước khi thấy data. Cho nên prior cho phép chúng ta khám phá các hệ quả khi khởi đầu bằng thông tin khác nhau. Trong trường hợp ta có thông tin prior tốt về tính phù hợp của các tham số, như mối quan hệ âm giữa height và weight, chúng ta có thể mã hoá thông tin đó trực tiếp vào prior. Khi không có đủ thông tin, chúng ta thường vẫn đủ biết được khoảng giá trị khả dĩ của nó. Và baạn có thể thay đổi prior và lặp lại phân tích để nghiên cứu xem ảnh hưởng của tình trạng thông tin ban đầu khác nhau đến suy luận như thế nào. Thông thường, có nhiều lựa chọn prior hợp lý, và chúng đều cho suy luận như nhau. Và các prior tiện lợi trong Bayes có tính chất <i>bảo tồn</i>, tương đương với cách tiếp cận non-Bayes tiện lợi. Chúng ta sẽ thấy tính chất bảo tồn này ở Chương 7.</p>
<p>Sự lựa chọn hay làm cho người mới lo lắng. Có một ảo tưởng rằng những quy trình mặc định là có hướng đối tượng tốt hơn những quy trình cần lựa chọn của người dùng, như chọn prior. Nếu điều đó đúng, thì tất cả "tính đối tượng" nghĩa là mọi người đều làm như nhau. Nó không đảm bảo mang tính thực tế hay chính xác.</p></div>

<div class="alert alert-dark">
<p><strong>Mô phỏng dự đoán prior và $p$-hacking.</strong> Một vấn nạn trong thống kê ứng dụng là "$p$-hacking", một hành vi chỉnh sửa mô hình và data để có được kết quả mong muốn. Kết quả mong muốn thường là $p$-value nhỏ 5%. Vấn đề là khi mô hình bị chỉnh sửa sau khi quan sát data, thì $p$-value không còn ý nghĩa gốc của nó nữa. Kết quả sai là phải được mong đợi. Chúng ta không quan tâm đến $p$-value trong sách này. Nhưng nguy hiểm vẫn còn đó, nếu chúng ta chọn prior đặt điều kiện trên các mẫu quan sát, chỉ để có được kết quả mong muốn. Quy trình chúng ta vừa thực hiện là chọn prior đặt điều kiện lên kiến thức của biến số trước khi thấy data - những ràng buộc, khoảng giá trị, mối quan hệ theo lý thuyết. Đó là lý do data chưa xuất hiện ở phần trước. Ta chọn prior dựa vào sự thật tổng quát, không phải từ mẫu. Chúng ta sẽ đánh giá hiệu năng của mô hình với data thực tiếp theo.</p></div>

### 4.4.2 Tìm phân phối posterior

Đoạn code cần để ước lượng posterior là đoạn code mà bạn đã thấy và được tuỳ chỉnh thêm một ít. Tất cả những gì chúng ta cần là thêm mô hình trung bình mới vào trong hàm định nghĩa mô hình, và đừng quên thêm prior mới, $\beta$. Hãy lặp lại định nghĩa mô hình, với dòng code tương ứng ở bên phải:

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu_i, \sigma) &&\quad {\Tiny numpyro.sample("height", dist.Normal(mu, sigma), obs=height)}\\
\mu_i &=\alpha + \beta(x_i - \bar{x}) &&\quad  {\Tiny mu = a + b * (weight - xbar)}\\
\alpha &\sim \text{Normal}(178, 20) &&\quad {\Tiny a = numpyro.sample("a", dist.Normal(178, 20))}\\
\beta &\sim \text{Log-Normal}(0, 10) &&\quad  {\Tiny b = numpyro.sample("b", dist.LogNormal(0, 1))}\\
\sigma &\sim \text{Uniform}(0, 50) &&\quad  {\Tiny sigma = numpyro.sample("sigma", dist.Uniform(0, 50))}\\
\end{aligned}$$

<b>code 4.42</b>
```python
# load data again, since it's a long way back
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv?raw=true", sep=";")
d2 = d[d["age"] >= 18]
# define the average weight, x-bar
xbar = d2.weight.mean()
# fit model
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b * (weight - xbar))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
m4_3 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m4_3, optim.Adam(1), Trace_ELBO(), weight=d2.weight.values,height=d2.height.values
)
p4_3, losses = svi.run(random.PRNGKey(0), 2000)
```

<div class="alert alert-info">
<p><strong>Mọi thứ phụ thuộc vào tham số đều có phân phối posterior.</strong> Trong mô hình ở trên, tham số $\mu$ không còn là tham số, vì nó trở thành hàm của tham số $\alpha$ và $\beta$. Nhưng vì tham số $\alpha$ và $\beta$ có posterior kết hợp, nên $\mu$ cũng vậy. Trong phần sau của chương này, bạn sẽ làm việc trực tiếp với phân phối posterior của $\mu$, mặc dù nó không còn là tham số nữa.  Bởi vì tham số là bất định, mọi thứ phụ thuộc vào nó đều bất định. Điều này bao gồm đại lượng thống kê như $\mu$, cũng như các dự đoán của mô hình, đo đạc mức độ fit, và mọi thứ khác có sử dụng tham số. Bằng cách lấy mẫu từ posterior, việc bạn cần làm là đưa tính bất định vào đại lượng cần dùng, để tính toán đại lượng với mỗi mẫu từ posterior. Kết quả là đại lượng đó, mỗi một cho từng mẫu posterior, sẽ tương đương phân phối posterior của nó.</p></div>

<div class="alert alert-dark">
<p><strong>Log và exp.</strong> Rất nhiều nhà khoa học tự nhiên và xã hội đã quên những gì họ biết về logarith một cách tự nhiên. Logarith xuất hiện rất nhiều trong thống kê ứng dụng. Bạn có thể nghĩ $y=\log(x)$ là gán $y$ cho mức độ lớn của $x$. Hàm $x=\exp(y)$ thì ngược lại, biến mức độ lớn thành giá trị.  Những định nghĩa này sẽ làm cho nhà toán học khoảng sợ. Nhưng rất nhiều công việc tính toán trong máy tính phụ thuộc vào những khái niệm này.</p>
<p>Định nghĩa này cho phép prior Log-Normal cho $\beta$ được mã hoá bằng cách khác. Thay vì định nghĩa tham số $\beta$, chúng ta định nghĩa một tham số mới là logarith của $\beta$ và gán nó vào phân phối normal. Sau đó chúng ta có thể đảo ngược logarith lại trong mô hình tuyến tính. Nó trông giống như vậy:</p>
{% highlight python %}
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    log_b = numpyro.sample("log_b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + jnp.exp(log_b) * (weight - xbar)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
{% endhighlight %}
<p> Để ý có <code>jnp.exp(log_b)</code> trong định nghĩa của <code>mu</code>. Nó là chung một mô hình với <code>m4_3</code>. Nó sẽ cho dự đoán giống nhau. Nhưng thay vì $\beta$ trong phân phối posterior, bạn được $\log(\beta)$. Việc chuyển đổi giữa chúng khá đơn giản, bởi vì $\beta=\exp(\log(\beta))$. Ở dạng code: <code>b = jnp.exp(log_b)</code>.</p></div>

### 4.4.3 Diễn giải phân phối posterior

Vấn đề của các mô hình thống kê là nó khó hiểu. Sau khi fit mô hình, nó chỉ có thể báo cáo lại phân phối posterior. Đó là câu trả lời đúng cho câu hỏi được bạn đặt ra. Nhưng trách nhiệm của bạn là xử lý câu trả lời đó và tìm hiểu nó.

Có hai cách xử lý chính: (1) đọc bảng (2) vẽ biểu đồ mô phỏng. Với câu hỏi đơn giản, ta có thể hiểu được posterior chỉ từ bảng giá trị biên. Nhưng phần lớn mô hình rất khó hiểu chỉ từ các bảng và con số. Khó khăn lớn nhất của bảng là tính đơn giản của nó so với mức độ phức tạp của mô hình cũng như data tạo ra nó. Một khi bạn có nhiều hơn vài tham số trong mô hình, nó rất khó để hình dung từ các con số, cách tương tác giữa tất cả những con số đó để ảnh hương dự đoán như thế nào. Nó cũng là lý do chúng ta mô phỏng từ prior. Một khi bạn bắt đầu thêm số hạng tương tác (Chương 8) hay đa thức (phần sau của chương), việc đoán mò sức ảnh hưởng của biến dự đoán lên biến kết cục là không khả thi.

Cho nên xuyến suốt sách này, tôi nhấn mạnh vẽ biểu đồ phân phối posterior và dự đoán posterior, thay vì cố gắng đọc hiểu các bảng. Vẽ đồ hoạ của những kết luận rút ra từ mô hình sẽ cho phép bạn điều tra những thứ mà khó đọc được từ các bảng:
1. Quy trình fit mô hình của hoạt động đúng hay chưa
2. Mức độ lớn *tuyệt đối*, thay vì mức độ lớn *tương đối*, của mối quan hệ giữa kết cục và biến dự đoán
3. Tính bất định xung quanh mối quan hệ trung bình
4. Tính bất định xung quanh dự đoán suy ra từ mô hình, vì chúng khác biệt với tính bất định của tham số

Thêm vào đó, một khi bạn rành hơn các thao tác vẽ phân phối posterior, bạn có thể đặt bất kỳ câu hỏi nào bạn nghĩ ra được, cho bất kỳ mô hình nào. Và người đọc kết quả của bạn sẽ rất biết ơn những biểu đồ đó hơn là những bảng giá trị ước lượng.

Cho nên trong phần còn lại, tôi trước tiên sẽ bỏ ít thời gian nói về các bảng ước lượng. Sau đó tôi đi tiếp để cho bạn thấy cách vẽ biểu đồ ước lượng mà có lồng ghép thông tin từ toàn bộ phân phối posterior, bao gồm tương quan giữa các tham số.

<div class="alert alert-info">
<p><strong>Tham số là gì></strong> Một vấn đề cơ bản của việc diễn giải ước lượng từ mô hình là hiểu được ý nghĩa của tham số. Không có một hướng dẫn cụ thể về ý nghĩa của tham số, tuy nhiên, bởi vì có nhiều người khác nhau có triết lý khác nhau về mô hình, xác suất và dự đoán. Góc nhìn trong sách này là một góc nhìn Bayes phổ biến: <i>Xác suất posterior của giá trị tham số mô tả tình phù hợp tương đối của trạng thái khác nhau của thế giới với data, theo như mô hình.</i> Đó là các con số trong thế giới nhỏ (Chương 2). Cho nên đương nhiên là nhiều người sẽ bất đồng về ý nghĩa trong thế giới lớn, và chi tiết của những bất đồng đó phụ thuộc mạnh vào bối cảnh. Những quan điểm bất đồng đó có thể là tốt, vì chúng dẫn đến đánh giá và cải thiện mô hình, thứ mà golem không tự làm được. Trong chương sau, bạn sẽ thể tham số có thể ám chỉ cho đại lượng quan sát được - data - cũng như những giá trị khong quan sát được. Điều này làm cho tham số hữu dụng hơn và diễn giải của chúng càng phụ thuộc hơn và bối cảnh.</p></div>

#### 4.4.3.1 Bảng của các phân phối biên.

Sau khi fit Kalahari data vào mô hình hồi quy tuyến tính, chúng ta có thể kiểm tra phân phối posterior biên của từng tham số.

<b>code 4.44</b>
```python
samples = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
print_summary(samples, 0.89, False)
```
<samp>
           mean       std    median      5.5%     94.5%     n_eff     r_hat
    a    154.62      0.27    154.63    154.16    155.03    931.50      1.00
    b      0.91      0.04      0.90      0.84      0.97   1083.74      1.00
sigma      5.08      0.19      5.08      4.79      5.41    949.65      1.00</samp>

Dòng đầu tiên là ước lượng quadratic approximation cho $\alpha$, dòng thứ hai cho $\beta$, và dòng thứ ba cho $\sigma$.

Hãy tập trung vào `b` ($\beta$), bởi vì nó là tham số mới. Bởi vì $\beta$ là một *slope*, giá trị 0.91 có thể được đọc là *khi một người nặng hơn 1 kg thì được mong đợi cao thêm 0.91 cm*. 89% của xác suất posterior nằm ở giữa 0.84 và 0.97. Điều này cho thấy những giá trị $\beta$ gần zero hoặc lớn hơn một đều không phù hợp với data và mô hình này. Chắc chắn nó không phải bằng chứng rằng mối quan hệ giữa chiều cao và trọng lượng là tuyến tính, bởi vì mô hình chỉ xem xét các đường thẳng. Nó chỉ nói rằng, nếu bạn chọn đường thẳng, thì đường thẳng với độ dốc (slope) quanh 0.9 là phù hợp nhất.

Nhớ rằng, con số trong bảng trên là không đủ để mô tả toàn bộ posterior bậc hai. Để làm chuyện đó, chúng ta cần thêm ma trận phương sai - hiệp phương sai (variance - covariance matrix). Bạn có thể thấy  hiệp phương sai giữa các tham số qua:

<b>code 4.45</b>
```python
vcov = jnp.round(jnp.cov(jnp.array(list(post.values()))), )
```
<samp>          a     b sigma
a     0.073 0.000 0.000
b     0.000 0.002 0.000
sigma 0.000 0.000 0.037</samp>

Có rất ít hiệp phương sai giữa các tham số trong trường hợp này. Sử dụng `az.plot_pair` sẽ hiển thị cả hai posterior biên và hiệp phương sai. Trong các phần bài tập ở cuối chương, bạn sẽ thấy sự thiếu hụt hiệp phương sai giữa các tham số có thể do nguyên nhân **TẬP TRUNG (CENTERING)**.

#### 4.4.3.2 Vẽ suy luận posterior với data

Vẽ suy luận posterior với data luôn luôn có ích hơn rất nhiều. Nó không chỉ giúp ta diễn giải posterior, mà còn kiểm tra giả định của mô hình. Khi mà suy luận của mô hình khác xa với mẫu quan sát được, thì bạn có thể nghi ngờ mô hình fit chưa tốt hoặc được thiết kế sai. Nhưng ngay cả nếu bạn chỉ dùng biểu đồ để diễn giải posterior, chúng là những công cụ quý báu. Với mô hình đơn giản như này, chúng ta có thể đọc kết quả từ bảng các con số để hiểu mô hình (nhưng không phải lúc nào cũng dễ). Nhưng với mô hình phức tạp hơn, đặc biệt là mô hình chứa hiệu ứng tương tác (interaction) (Chương 8), diễn giải phân phối posterior rất khó. Kết hợp vào thêm vấn đề đưa thông tin hiệp phương sai vào diễn giả của bạn, biểu đồ là không thể thiếu được.

Chúng ta sẽ bắt đầu với một phiên bản đơn giản của tác vụ này, chỉ dùng giá trị trung bình của posterior vào data chiều cao và trọng lượng. Sau đó chúng ta sẽ từ từ thêm nhiều và nhiều thông tin hơn vào biểu đồ dự đoán, cho đến khi chúng ta dùng toàn bộ phân phối posterior.

Chúng ta sẽ bắt đầu chỉ với data thô và một đường thẳng. Đoạn code dưới sẽ thể hiện data thô, tính giá trị trung bình trong posterior của `a` và `b`, sau đó vẽ đường thẳng được suy ra từ chúng:

<b>code 4.46</b>
```python
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"))
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
a_map = np.mean(post["a"])
b_map = np.mean(post["b"])
x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
plt.plot(x, a_map + b_map * (x - xbar), "k");
```

<a name="f6"></a>![](/assets/images/fig 4-6.svg)
<details><summary>Hình 4.6: Chiều cao theo centimet (trục tung) được vẽ đối với trọng lượng theo kilogram (trục hoành), với đường thẳng ở trung bình posterior màu đen.</summary>
{% highlight python %}az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"))
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
a_map = np.mean(post["a"])
b_map = np.mean(post["b"])
x = np.linspace(d2.weight.min(), d2.weight.max(), 101)
plt.plot(x, a_map + b_map * (x - xbar), "k"){% endhighlight %}
</details>

Bạn có thể kết quả ở [**HÌNH 4.6**](#f6). Mỗi điểm trong biểu đồ này là một mẫu quan sát riêng lẻ. Đường màu đen được định nghĩa bởi trung bình của slope $\beta$ và trung bình của intercept $\alpha$. Đường thẳng này không tệ. Nó khá là phù hợp. Nhưng cũng có vô số các con đường thẳng phù hợp khác quanh nó. Chúng ta hãy vẽ chúng.

#### 4.4.3.3 Thêm tính bất định quanh trung bình

Đường thẳng trung bình posterior chỉ là trung bình của posterior, là đường thẳng có tính phù hợp cao nhất trong vô số đường thẳng khả dĩ của phân phối posterior. Biểu đồ đường trung bình, như [**HÌNH 4.6**](#f6), là hữu ích để gây ấn tượng cho mức độ của suy luận được ước lượng của một biến số. Nhưng nó rất kém để truyền tải tính bất định. Nhớ rằng, phân phối posterior xem xét mọi đường thẳng hồi quy nối chiều cao đến trọng lượng. Nó gán tính phù hợp tương đối cho mỗi đuòng thẳng. Điều này có nghĩa mỗi cặp $\alpha$ và $\beta$ đều có xác suất posterior. Có thể có rất nhiều đường thẳng có xác suất posterior gần giống như đường trung bình. Hoặc cũng có thể thay vào đó là phân phối posterior thì hẹp xung quanh đường trung bình.

Vậy làm sao để chúng ta thêm tính bất định vào biểu đồ? Kết hợp lại, cặp $\alpha$ và $\beta$ định nghĩa một đường thẳng. Và cho nên chúng ta có thể lấy mẫu nhiều đường thẳng từ phân phối posterior. Sau đó chúng ta có thể hiện thị những đường đó lên biểu đồ, để hiển thị đồ hoạ tính bất định vào trong mối quan hệ hồi quy.

Để nhận biết tốt hơn tại sao phân phối posterior bao gồm các đường thẳng, chúng ta sẽ làm việc với tất cả các mẫu từ mô hình. Hãy nhìn kỹ vào các mẫu từ posterior:

<b>code 4.47</b>
```python
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
{param: list(post[param].reshape(-1)[:5]) for param in post}
```
<samp>         a         b    sigma
1 154.5505 0.9222372 5.188631
2 154.4965 0.9286227 5.278370
3 154.4794 0.9490329 4.937513
4 155.2289 0.9252048 4.869807
5 154.9545 0.8192535 5.063672</samp>

Mỗi dòng là các mẫu ngẫu nhiên có tương quan từ phân phối kết hợp của cả ba tham số, bằng cách sử dụng hiệp phương sai. Cặp giá trị của `a` và `b` ở mỗi dòng tương ứng với một đường thẳng. Trung bình của rất nhiều đường thẳng như vậy là đường trung bình của posterior. Nhưng những đường phân tán quanh trung bình cũng có ý nghĩa, bởi vì nó thay đổi độ tin cậy của chúng ta về mối quan hệ giữa biến dự đoán và biến kết cục.

Cho nên bây giờ ta sẽ thể hiện những đường thẳng này, để bạn thấy sự phân tán. Bài học này sẽ dễ hiểu hơn, nếu chúng ta chỉ dùng vài data để bắt đầu. Sau đó bạn sẽ thấy khi thêm nhiều data vào thì sẽ thay đổi sự phân tán các đường thẳng. Cho nên chúng ta sẽ bắt đầu chỉ với 10 trường hợp đầu tiên ở `d2`. Code dưới đây sẽ lấy 10 trường hợp đầu tiên và tái ước lượng mô hình:

<b>code 4.48</b>
```python
N = 10
dN = d2[:N]
def model(weight, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + b * (weight - jnp.mean(weight))
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
mN = AutoLaplaceApproximation(model)
svi = SVI(
    model, mN, optim.Adam(1), Trace_ELBO(), weight=dN.weight.values, height=dN.height.values
)
pN, losses = svi.run(random.PRNGKey(0), 1000)
```

Bây giờ hãy vẽ 20 đường thẳng này, để xem tính bất định nhìn như thế nào.

<b>code 4.49</b>
```python
# extract 20 samples from the posterior
post = mN.sample_posterior(random.PRNGKey(1), pN, (20,))
# display raw data and sample size
ax = az.plot_pair(dN[["weight", "height"]].to_dict(orient="list"))
ax.set(
    xlim=(d2.weight.min(), d2.weight.max()),
    ylim=(d2.height.min(), d2.height.max()),
    title="N = {}".format(N),
)
# plot the lines, with transparency
x = jnp.linspace(d2.weight.min(), d2.weight.max(), 101)
for i in range(20):
    plt.plot(x, post["a"][i] + post["b"][i] * (x - dN.weight.mean()), "k", alpha=0.3)
```

Dòng cuối cùng sẽ lặp lại trên 20 đường thẳng, dùng `plt.plot` để vẽ từng đường.

<a name="f7"></a>![](/assets/images/fig 4-7.svg)
<details class="fig"><summary>Hình 4.7: Mẫu từ phân phối posterior được ước lượng quadratic approximation cho mô hình chiều cao/trọng lượng, <code>m4_3</code>, với số lượng data tăng dần lên. Với mỗi biểu đồ, 20 đường thẳng được lấy từ phân phối posterior, thể hiện cho tính bất định trong mối quan hệ hồi quy.</summary>
{% highlight python %}fig, axs = plt.subplots(2, 2, figsize=(10,8))
for ax, n in zip(axs.flatten(), [10,50,150,352]):
    dN = d2[:n]
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(),
              weight=dN.weight.values, height=dN.height.values)
    params, losses = svi.run(random.PRNGKey(0), 2000)
    post = guide.sample_posterior(random.PRNGKey(1), params, (20,))
    # display raw data and sample size
    az.plot_pair(dN[["weight", "height"]].to_dict("list"), ax=ax)
    ax.set(
        xlim=(d2.weight.min(), d2.weight.max()),
        ylim=(d2.height.min(), d2.height.max()),
        title="N = {}".format(n)
    )
    # plot the lines, with transparency
    x = np.linspace(d2.weight.min(), d2.weight.max(), 20)
    for i in range(20):
        ax.plot(x, post["a"][i] + post["b"][i] * (x - dN.weight.mean()),
                     "C1", alpha=0.1)
plt.tight_layout(){% endhighlight %}</details>

Kết quả được thể hiện ở biểu đồ trên bên trái trong [**HÌNH 4.7**](#f7) Bằng cách vẽ nhiều đường thẳng hồi quy, được lấy mẫu từ posterior, chúng ta dễ dàng nhìn ra cả hai những khía cạnh tin cậy cao của mối quan hệ và những khía cạnh tin cậy thấp hơn. Đám mây các đường hồi quy cho thấy mức độ bất định cao hơn ở các giá trị cực của trọng lượng.

Những biểu đồ khác trong [**HÌNH 4.7**](#f7) cho thấy mối quan hệ tương tự, nhưng với số lượng data tăng dần lên. Chỉ bằng việc dùng lại code cũ, nhưng thay đổi `N = 10` thành giá trị khác. Chú ý rằng đám mây các đường hồi quy tập trung lại hơn khi cỡ mẫu tăng lên. Điều này là kết quả của việc mô hình ngày càng tin cậy hơn với vị trí của trung bình.

#### 4.4.3.4 Vẽ khoảng và contour hồi quy

Đám mây đường hồi quy trong [**HÌNH 4.7**](#f7) nhìn rất đẹp, bởi vì nó thể hiện tính bất định của mối quan hệ theo cách mà mọi người đều cảm thấy dễ hiểu. Nhưng có phương pháp phổ biến và trực quan hơn để nhìn thấy tính bất định, đó là vẽ khoảng hay contour xung quanh đường hồi quy trung bình. Trong phần này, tôi sẽ hướng dẫn bạn cách tính khoảng bất kỳ mà bạn thích, thông qua đám mây đường hồi quy nằm trong phân phối posterior.

Bây giờ hãy tập trung vào một giá trị `weight`, ví dụ như 50 Kg. Bạn có thể rất nhanh tạo ra một danh sách 10,000 giá trị $\mu$ cho một cá nhân nặng 50 kg, thông qua các mẫu lấy từ posterior:

<b>code 4.50</b>
```python
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
mu_at_50 = post["a"] + post["b"] * (50 - xbar)
```

Ở dạng biểu thức cho $\mu_i$:

$$ \mu_i = \alpha + \beta (x_i - \bar{x})$$

Giá trị $x_i$ trong trường hợp này là 50. Bạn hãy thử nhìn vào kết quả, `mu_at_50`. Nó là một vector các con số trung bình dự đoán, một cái cho từng mẫu ngẫu nhiên từ posteriỏr. Bởi vì `a` và `b` đều đưa vào trong phép tính, sự biến thiên xuyên suốt những trung bình này đã sáp nhập tính bất định trong và tương quan giữa cả hai tham số. Thời điểm thì thích hợp để vẽ ra mật độ của vector các trung bình này:

<b>code 4.51</b>
```python
az.plot_kde(mu_at_50, label="mu|weight=50")
```

<a name="f8"></a>![](/assets/images/fig 4-8.svg)
<details class="fig"><summary>Hình 4.8: Ước lượng quadratic approximation cho phân phối posterior của chiều cao trung bình, $\mu$, khi trọng lượng là 50 kg. Phân phối này đại diện cho tính phù hợp tương đối của giá trị khác nhau của trung bình.</summary>
{% highlight python %}post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
mu_at_50 = post["a"] + post["b"] * (50 - xbar)
az.plot_kde(mu_at_50, bw=0.05)
plt.gca().set(xlabel="mu|weight=50", ylabel="mật độ"){% endhighlight %}</details>

Tôi đã dựng lại biểu đồ này ở [**HÌNH 4.8**](#f8). Bởi vì thành phần của $\mu$ có phân phối, nên $\mu$ cũng vậy. Và bởi vì phân phối của $\alpha$ và $\beta$ là Gaussian, nên phân phối của $\mu$ cũng vậy (Cộng các phân phối Gaussian lại luôn tạo ra một phân phối Gaussian).

Bởi vì posterior của $\mu$ là phân phối, bạn có thể tìm các khoảng của nó, giống như trong phân phối posterior bất kỳ. Để tìm khoảng tin cậy 89% của $\mu$ tại 50 kg, thì chỉ cần dùng hàm `jnp.quantile` hoặc `jnp.percentile`.

<b>code 4.52</b>
```python
jnp.percentile(mu_at_50, q=(5.5, 94.5))
```
<samp>[158.5957 , 159.71445]</samp>

Hai con số này nghĩa là khoảng trung tâm 89% cho các cách mà mô hình tạo ra data đặt chiều cao trung bình khoảng 159cm và 160cm (đặt điều kiện trên mô hình và data), giả định tại trọng lượng là 50 kg.

Mọi thứ đều diễn ra tốt đẹp, nhưng chúng ta cần lặp lại phép tính trên cho toàn bộ giá trị `weight` ở trục hoành, chứ không chỉ khi nó là 50 kg. Chúng ta muốn vẽ khoảng 89% xung quanh slope trung bình trong [**HÌNH 4.6**](#f6).

Điều này được thực hiện thông qua hàm `Predictive` trong `numpyro`. Nó nhận mô hình và mẫu rút ra từ posterior để thực hiện dự đoán, bạn có thêm dùng những mẫu quan sát từ data hoặc những data mới. Trong trường hợp này thì chúng ta dùng lại data được fit vào mô hình.

<b>code 4.53</b>
```python
mu = Predictive(m4_3.model, post, return_sites=["mu"])(
    random.PRNGKey(2), d2.weight.values, d2.height.values
)["mu"]
mu.shape, list(mu[:5, 0])
```
<samp>((1000, 352), [157.12938, 157.30838, 157.05736, 156.90125, 157.4044])</samp>

Bạn sẽ có được một ma trận các giá trị $\mu$. Mỗi dòng là một mẫu từ phân phối posterior. Bạn đã rút ra 1000 mẫu từ posterior và đã gán chúng vào `post`. Mỗi cột là một trường hợp (dòng) trong data. Có 352 dòng trong `d2`, tương ứng 352 cá thể. Cho nên có 352 cột trong ma trận `mu` ở trên.

Chúng ta có thể làm được gì với ma trận khổng lồ này? Rất nhiều thứ. Chúng ta có phân phối $\mu$ cho mỗi cá thể trong data gốc. Chúng ta thực ra muốn muốn thứ hơi khác: một phân phối $\mu$ cho mỗi giá trị trọng lượng độc nhất trên trục hoành. Tính toán nó thì hơi khó hơn một chút.

<b>code 4.54</b>
```python
# define sequence of weights to compute predictions for
# these values will be on the horizontal axis
weight_seq = jnp.arange(start=25, stop=71, step=1)

# use predictive to compute mu
# for each sample from posterior
# and for each weight in weight_seq
mu = Predictive(m4_3.model, post, return_sites=["mu"])(
    random.PRNGKey(2), weight_seq, None
)["mu"]
mu.shape, list(mu[:5, 0])
```
<samp>((1000, 46), [134.88252, 136.99348, 138.36269, 137.87814, 134.30676])</samp>

Và bây giờ ta có chỉ 46 cột trong `mu`, bởi vì ta chỉ cho vào 46 giá trị `weight` khác nhau. Để biểu diễn những gì bạn có, ta vẽ biểu đồ phân phối của giá trị $\mu$ tại mỗi chiều cao.

<b>code 4.55</b>
```python
# use scatter_kwargs={"alpha": 0} to hide raw data
az.plot_pair(
    d2[["weight", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0}
)
# loop over samples and plot each mu value
for i in range(100):
    plt.plot(weight_seq, mu[i], "o", c="royalblue", alpha=0.1)
```

<a name="f9"></a>![](/assets/images/fig 4-9.svg)
<details class="fig"><summary>Hình 4.9: Trái: 100 giá trị đầu tiên trong phân phối của $\mu$ tại mỗi giá trị trọng lượng. Phải: Cũng là data chiều cao người !Kung, bây giờ thêm khoảng tin cậy 89% của trung bình là vùng được tô màu. Hãy so sánh vùng này với phân phối của các điểm xanh bên trái.</summary>
{% highlight python %}weight_seq = jnp.arange(start=25, stop=71, step=1)
fig, axs = plt.subplots(1,2, figsize=(12,5))
mu = Predictive(m4_3.model, post, return_sites=["mu"])(
    random.PRNGKey(2), weight_seq, None
)["mu"]
for i in range(100):
    axs[0].plot(weight_seq, mu[i], "o", c="royalblue", alpha=0.1)
axs[1].scatter(d2['weight'], d2['height'], alpha=0.5)
shaded = jnp.quantile(mu , q=jnp.array([0.055, 0.945]), axis=0)
axs[1].fill_between(weight_seq, shaded[0], shaded[1], color="C2", alpha=0.5)
axs[1].plot(weight_seq, jnp.mean(mu,axis=0),color="C1", linewidth=1)
for ax in axs:
    ax.set(xlabel="trọng lượng", ylabel="chiều cao")
plt.tight_layout(){% endhighlight %}
</details>

Kết quả được hiện lên bên trái của [**HÌNH 4.9**](#f9). Tại mỗi giá trị trọng lượng ở `weight_seq`, một tập hợp các giá trị $\mu$ được hiển thị. Mỗi tập hợp này là một phân phối Gaussian, như trong [**HÌNH 4.8**](#f8). Bạn có thể thấy là mức độ bất định tại $\mu$ phụ thuộc vào giá trị của `weight`. Và đây là chung hiện tượng bạn nhìn thấy ở [**HÌNH 4.7**](#f7).

Bước cuối cùng là tóm tắt phân phối của mỗi giá trị trọng lượng.

<b>code 4.56</b>
```python
mu_mean = jnp.mean(mu, axis=0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
```

Dòng đầu tiên là tính trung bình của mỗi cột `(axis=0)` trong ma trận `mu`. Bây giờ `mu_mean` chứa các giá trị trung bình $\mu$ tại mỗi giá trị trọng lượng, và `mu_PI` chứa biên dưới và biên trên của khoảng 89% tại mỗi giá trị trọng lượng. Bạn nên kiểm tra lại `mu_mean` và `mu_PI` để rõ hơn. Chúng chỉ là các loại tóm tắt khác nhau về phân phối trong `mu`, với mỗi cột tương ứng cho một giá trị trọng lượng khác nhau. Những tóm tắt này cũng chỉ là tóm tắt. Còn "ước lượng" là toàn bộ phân phối.

Bạn có thể vẽ biểu đồ tóm tắt trên data chỉ với vài dòng code:

<b>code 4.57</b>
```python
# plot raw data
# fading out points to make line and interval more visible
az.plot_pair(
    d2[["weight", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5}
)
# plot the MAP line, aka the mean mu for each weight
plt.plot(weight_seq, mu_mean, "k")
# plot a shaded region for 89% PI
plt.fill_between(weight_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
```

Kết quả nằm ở bên phải của [**HÌNH 4.9**](#f9).

Bằng phương pháp này, bạn có thể suy ra và vẽ trung bình và khoảng tin cậy của dự đoán posterior cho những mô hình phức tạp, với bất kỳ data mà bạn chọn. Mặc dù chúng ta có thể dùng phân tích toán học để tính những khoảng như vậy. Tôi đã từng giảng dạy cách tiếp cận bằng phân tích toán học trước đây, và nó là một thảm hoạ. Một trong những lý do có lẽ là tôi là một giáo viên thất bại, nhưng phần còn lại là đa số nhà khoa học và xã hội chưa bao giờ có nhiều kinh nghiệm về thuyết xác suất và có khuynh hướng lo lắng khi gặp dấu $\int$. Tôi chắc chắn là với đủ cố gắng, mọi người trong họ sẽ học được cách làm toán. Nhưng toàn bộ họ sẽ học nhanh hơn các kỹ thuật tạo mẫu và tóm tắt mẫu được rút ra từ phân phối posterior. Cho nên trong khi toán học là một cách tiếp cận tinh tế, và nó có những ý nghĩa đáng giá đến từ hiểu biết các phép tính, cách tiếp cận dân gian được trình bày ở đây là rất linh hoạt và cho phép một lượng lớn nhà khoa học tiếp cận để rút ra những hiểu biết từ việc thiết kế mô hình thống kê. Và lần nữa, khi bạn bắt đầu ước lượng bằng MCMC (Chương 9), thì đây là cách tiếp cận duy nhất. Cho nên nó đáng để học bây giờ.

Tóm lại, đây là công thức để tạo ra các dự đoán và khoảng từ posterior của mô hình được fit.
1. Sử dụng `Predictive` để tạo các giá trị posterior cho $\mu$. Ta có thể cho vào data gốc, hoặc bạn phải cho một dãy các giá trị có trục hoành nếu bạn muốn vẽ dự đoán posterior.
2. Dùng hàm như `jnp.mean` và `jnp.quantile` để tìm trung bình và biên dưới và biên trên cho $\mu$ cho mỗi giá trị của biến dự đoán.
3. Cuối cùng, sử dụng các chức năng đồ hoạ như `plt.plot`, `plt.fill_between` để vẽ đường thẳng và khoảng. Hoặc bạn có thể vẽ phân phối các dự đoán, hoặc xa hơn là tính toán số học với chúng. Nó thực sự tuỳ vào bạn.

Công thức này hoạt động tốt cho mọi mô hình chúng ta sẽ học trong sách này. Chỉ cần bạn biết cấu trúc của mô hình - liên quan giữa tham số và data - bạn có thể sử dụng các mẫu từ posterior để mô tả bất kỳ khía cạnh nào của hành vi của mô hình.

<div class="alert alert-info">
<p><strong>Khoảng quá tin cậy.</strong> Khoảng tin cậy cho đường hồi quy trong [**HÌNH 4.9**](#f9) thì bám chắc xung quanh đường MAP. Nghĩa là có rất ít tính bất định về chiều cao trung bình như là một hàm của trọng lượng trung bình. Nhưng bạn phải ghi nhớ rằng những suy luận này luôn được đặt điều kiện trên mô hình. Ngay cả một mô hình tệ cũng có thể có khoảng tin cậy rất chắc. Sẽ tốt hơn nếu bạn nghĩ đường hồi quy trong [**HÌNH 4.9**](#f9) như nói rằng: <i>Điều kiện trên giả định là chiều cao và trọng lượng liên quan với nhau bằng một đường thẳng, thì đây là đường thẳng khả dĩ nhất, và chúng là biên phù hợp của nó.</i></p></div>

<div class="alert alert-dark">
<p><strong><code>Predictive</code> hoạt động như thế nào.</strong> Hàm <code>Predictive</code> không có quá phức tạp. Tất cả những gì nó thực hiện là sử dụng hàm công thức mô hình của bạn cung cấp khi bạn fit mô hình để tính giá trị của mô hình tuyến tính. Nó làm điều này cho tất cả các mẫu từ phân phối posterior, cho mỗi giá trị của biến dự đoán. Bạn có thể hoàn thành việc này cho bất kỳ mô hình nào, fit bằng bất kỳ cách nào, bằng tự mình thao tác. Với <code>m4_3</code> thì nó như vậy:</p>
<b>code 4.58</b>
{% highlight python %}post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
mu_link = lambda weight: post["a"] + post["b"] * (weight - xbar)
weight_seq = jnp.arange(start=25, stop=71, step=1)
mu = vmap(mu_link)(weight_seq).T
mu_mean = jnp.mean(mu, 0)
mu_HPDI = hpdi(mu, prob=0.89, axis=0){% endhighlight %}
<p>Và các giá trị trong <code>mu_mean</code> và <code>mu_HDPI</code> sẽ rất giống (cho phép sự biến thiên mô phỏng) với những gì bạn có được từ <code>Predictive</code>.</p>
<p>Biết được cách hoạt động này là hữu ích cho việc (1) hiểu và (2) sức mạnh. Cho dù bạn dùng mô hình nào, cách tiếp cận này có thể dùng để tạo dự đoán posterior cho bất kỳ thành phần nào của nó. Công cụ tự động như <code>Predictive</code> sẽ giúp tiết kiệm thời gian, nhưng chúng không bao giờ linh hoạt hơn code mà bạn tự viết.</p>
</div>

#### 4.4.3.5 Khoảng dự đoán

Bây giờ chúng ta sẽ tạo ra khoảng dự đoán 89% cho chiều cao thực tế, chứ không chỉ chiều cao trung bình, $\mu$. Điều này có nghĩa chúng ta sẽ bao gồm độ lệch chuẩn $\sigma$ và tính bất định của nó. Nhớ lại rằng, dòng đầu tiên của mô hình thống kê là:

$$ h_i \sim \text{Normal}(\mu_i, \sigma)$$

Chúng ta đã sử dụng mẫu từ posterior để thể hiện trên biểu đồ về tính bất định của $\mu_i$, mô hình tuyến tính của trung bình. Nhưng dự đoán thực tế của chiều cao phụ thuộc vào phân phối ở dòng đầu tiên. Phân phối Gaussian ở dòng đầu tiên nói cho chúng ta biết là mô hình mong đợi chiều cao quan sát được là phân phối xung quanh $\mu$, không phải ngay trên nó. Và độ toả ra xung quanh $\mu$ được kiểm soát vởi $\sigma$. Tất cả những thứ gì nói lên việc chúng ta phải cần kết hợp $\sigma$ vào trong dự đoán bằng một cách nào đó.

Cách làm như sau. Hãy tưởng tượng chúng ta đang mô phỏng các chiều cao. Với bất kỳ giá trị trọng lượng độc nhất, bạn lấy mẫu từ phân phối Gaussian với đúng giá trị trung bình $\mu$ cho trọng lượng đó, sử dụng đúng giá trị $\sigma$ được rút ra từ cùng phân phối posterior. Nếu bạn thực hiện động tác này cho tất cả mẫu từ posterior, cho mọi giá trị trọng lượng quan tâm, bạn sẽ có được một tập hợp các chiều cao được mô phỏng mà biểu hiện cho tính bất định trong posterior cũng như tính bất định trong phân phối Gaussian về chiều cao. Công cụ `Predictive` sẽ giúp chúng ta dễ dàng thực hiện điều này:

<b>code 4.59</b>
```python
sim_height = Predictive(m4_3.model, post, return_sites=["height"])(
    random.PRNGKey(2), weight_seq, None
)["height"]
sim_height.shape, list(sim_height[:5, 0])
```
<samp>((1000, 46), [135.85771, 137.52162, 133.89777, 138.14607, 131.1664])</samp>

Ma trận này cũng giống như ma trận trước, `mu`, nhưng nó chứa những chiều cao được mô phỏng, không phải phân phối của chiều cao trung bình phù hợp, $\mu$.

Chúng ta có thể tóm tắt những chiều cao mô phỏng này như cách chúng ta đã tóm tắt phân phối của $\mu$.

<b>code 4.60</b>
```python
height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
```

Bây giờ, `height_PI` chứa khoảng dự đoán posterior 89% cho các chiều cao quan sát được (theo như mô hình), xuyên suốt dãy giá trị các trọng lượng trong `weight_seq`.

Hãy vẽ biểu đồ thể hiện tất cả những gì chúng ta đã xây dựng: (1) đường trung bình, (2) vùng tô đậ 89% các $\mu$ phù hợp, và (3) biên giới của chiều cao mô phỏng theo mong đợi của mô hình.

<b>code 4.61</b>
```python
az.plot_pair(d2[["weight", "height"]].to_dict(orient="list"),
             plot_kwargs={"alpha": 0.5})

# draw MAP line
plt.plot(weight_seq, mu_mean, "k")

# draw HPDI region for line
plt.fill_between(weight_seq, mu_PI[0], mu_HPDI[1], color="k", alpha=0.2)

# draw PI region for simulated heights
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k",
                 alpha=0.15);
```

Đoạn code trên có sử dụng những đối tượng được tính ở phần trước, cho nên nếu cần bạn có thể thể quay lại và thực thi đoạn code đó.


<a name="f10"></a>![](/assets/images/fig 4-10.svg)
<details><summary>Hình 4.10: Khoảng dự đoán 89% cho chiều cao, như là hàm của trọng lượng. Đường nét liền là đường trung bình cho trung bình chiều cao tại mỗi điểm trọng lượng. Hai vùng được tô cho thấy khoảng tin cậy 89% khác nhau. Vùng tô màu nhỏ hơn xung quanh đường thẳng là phân phối của $\mu$. Vùng tô màu rộng hơn đại diện của vùng mà trong đó mô hình mong đợi tìm thấy 89% chiều cao thực tế trong quần thể, tại mỗi giá trị trọng lượng.</summary>
{% highlight python %}fig = plt.figure(figsize=(12,5))
height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
az.plot_pair(
    d2[["weight", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5})
plt.plot(weight_seq, mu_mean, "C0", linewidth=1)
plt.fill_between(weight_seq, mu_HPDI[0], mu_HPDI[1], color="C0", alpha=0.2)
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="C0", alpha=0.15)
plt.gca().set(xlabel="cân nặng", ylabel="chiều cao"){% endhighlight %}
</details>

Tôi đã vẽ kết quả trong [**HÌNH 4.10**](#f10). Vùng tô màu rộng trong hình đại diện cho khoảng 89% các chiều cao thực tế trong quần thể mà mô hình mong đợi. Không có gì đặc biệt về con số 89%. Bạn có thể vẽ biên giới với số bách phân vị khác, như 67% và 97% (cũng là số nguyên tố), và thêm chúng vào biểu đồ. Làm như vậy sẽ giúp bạn nhìn rõ hơn hình dángcủa phân phối chiều cao dự đoán. Tôi đã để chúng lại vào bài tập cho bạn đọc. Bạn chỉ cần quay về đoạn code trên và thay đổi `q=[16,83]`, ví dụ, vào hàm `jnp.percentile`. Nó sẽ cho bạn khoảng 67%, thay vì khoảng 89%.

Chú ý rằng đường viền của vùng tô màu rộng hơi gồ ghề. Đó là do biến thiên mô phỏng ở hai đuôi của giá trị rút ra từ Gaussian. Nếu thấy khó chịu, bạn có thể tăng thêm số lượng mẫu ra từ phân phối posterior. Tham số `sample_shape` trong hàm `sample_posterior` kiểm soát số lượng mẫu để sử dụng. Ví dụ:

<b>code 4.62</b>
```python
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, sample_shape=(int(1e4),))
sim_height = Predictive(m4_3.model, post, return_sites=["height"])(
    random.PRNGKey(2), weight_seq, None
)["height"]
height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
```

Chạy lại đoạn code vẽ biểu đồ trên, bạn sẽ thấy được biên giới của vùng tô màu đã được làm mượt. Với bách phân vị ở hai cực, có thể rất khó để làm mượt hoàn toàn. Rất may, nó không ảnh hưởng nhiều, ngoại trừ tính thẩm mỹ. Hơn nữa, nó ở đó để nhắc cho chúng ta rằng suy luận thống kê là tương đối. Sự thật chúng ta có thể tính được giá trị mong đợi ở hàng thập phân thứ 10 không phải tương đương suy luận của chúng ta là chính xác đến hàng thập phân thứ 10.

<div class="alert alert-info">
<p><strong>Hai loại tính bất định.</strong> Ở quy trình trên, chúng ta gặp cả tính bất định của tham số và tính bất định trong quá trình xử lý lấy mẫu. Đây là hai khái niệm khác nhau, mặc dù chúng được xử lý như nhau và cuối cùng được trộn lại trong mô phỏng dự đoán posterior. Phân phối posterior là các thứ hạng của tính phù hợp tương đối của mọi sự kết hợp các giá trị tham số. Phân phối của kết cục được mô phỏng, như chiều cao, thay vào đó là phân phối bao gồm sự biến thiên do lấy mẫu từ quy trình tạo ra các giá trị Gaussian ngẫu nhiên. Sự biến thiên do lấy mẫu này cũng là giá định của mô hình. Tính đối tượng của nó của không hơn kém gì so với phân phối posterior. Cả hai tính bất định đều quan trọng. Nhưng quan trọng hơn là phân biệt chúng, bởi vì chúng phụ thuộc vào giả định khác nhau. Hơn nữa, chúng ta có thể xem likelihood Gaussian như là một giả định về phương pháp học (một công cụ để ước lượng trung bình và phương sai của một biến số), hơn là một giả định về tự nhiên về data trong tương lai sẽ như thế nào. Trong trường hợp đó, chúng ta không thể diễn giải hoàn toàn các kết cục được mô phỏng.</p></div>

<div class="alert alert-dark">
<p><strong>Tự mình tạo ra mô phỏng.</strong> Thay vì dùng <code>Predictive</code>, bạn có thể dùng phân phối likelihood để mô phỏng ra các giá trị kết cục. Trong ví dụ này thì likelihood là Gaussian.</p>
{% highlight python %}
post = m4_3.sample_posterior(random.PRNGKey(1), p4_3, (1000,))
weight_seq = jnp.arange(25, 71)
sim_height = vmap(
    lambda i, weight: dist.Normal(post["a"] + post["b"] * (weight - xbar)).sample(
        random.PRNGKey(i)
    )
)(jnp.arange(len(weight_seq)), weight_seq)
height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
{% endhighlight %}
<p>Giá trị trong <code>height_PI</code> sẽ gần giống với giá trị được tính ở bài chính và được thể hiện trong <a href="#f10"><strong>HÌNH 4.10</strong></a>.</p></div>

## <center>4.5 Đường cong từ đường thẳng</center><a name="a5"></a>

Trong chương tiếp theo, bạn sẽ gặp mô hình hồi quy tuyến tính với nhiều hơn một biến dự đoán. Nhưng trước đó, bạn cũng nên xem cách thiết kê mô hình cho biến kết cục bằng hàm đường cong của một biến dự đoán. Tất cả mô hình cho đến hiện tại đều giả định quan hệ là một đường thẳng, Nhưng không có gì đặc biệt về đường thẳng, ngoài sự đơn giản của chúng.

Chúng ta sẽ xem xét hai phương pháp phổ thông sử dụng hồi quy tuyến tính để tạo đường cong. Một là **HỒI QUY ĐA THỨC (POLYNOMIAL REGRESSION)** và hai là **B-SPLINES**. Cả hai phương pháp đều hoạt động bằng sự biến đổi một biến dự đoán thành nhiều biến được tổng hợp. Nhưng spline thì có nhiều ưu thế hơn. Cả hai phương pháp đều không gì hơn ngoài mục đích mô tả một hàm số để liên quan một biến này đến mọt biến kia. Suy luận nhân quả, mà chúng ta sẽ học ở chương sau, muốn nhiều thứ hơn.

### 4.5.1 Hồi quy đa thức

Hồi quy đa thức dùng các bậc luỹ thừa của một biến - bình phương hoặc lập phương - làm biến dự đoán mới. Đây là phương pháp đơn giản để tạo quan hệ đường cong. Hồi quy đa thức rất phổ biến, và hiểu cách nó hoạt động sẽ giúp cho các mô hình về sau. Để tìm hiểu cách hoạt động của hồi quy đa thức, hãy thực hành qua ví dụ, bằng toàn bộ data !Kung, chứ không chỉ người lớn:

<b>code 4.64</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv?raw=true", sep=";")
```

Hãy nhìn vào `az.plot_pair` giữa chiều cao và trọng lượng. Mối quan hệ có thể thấy được là nó cong, bởi vì chúng ta đã thêm những cá thể không phải người lớn.

Hồi quy đa thức thông dụng nhất là mô hình parabol của trung bình. Gọi x là trọng lượng được chuẩn hoá. Thì phương trình bậc hai của trung bình chiều cao có dạng:

$$ \mu_i = \alpha + \beta_1x_i + \beta_2x_i^2 $$

Trên đây là đa thức parabol (bậc 2). Phần $\alpha + \beta_1x_i$ cũng giống như hàm tuyến tính của $x$ trong hồi quy tuyến tính, với con số "1" nhỏ dưới tên của tham số, để chúng ta có thể phân biệt được với tham số mới. Phần số hạng thêm vào dùng bình phường của $x_i$ để dựng lên một parabol, hơn là đường thẳng hoàn hảo. Tham số mới $\beta_2$ đo lường độ cong của mối quan hệ.

Fit data vào mô hình là dễ. Diễn giải chúng thì khó hơn. Chúng ta sẽ bắt đầu ở phần dễ trước, fit mô hình parabol của chiều cao trên trọng lượng. Việc đầu tiên cần làm là **CHUẨN HOÁ (STANDARDIZE)** biến dự đoán. Khi biến dự đoán có những giá trị rất lớn, đôi khi sẽ có lỗi liên quan đế con số. Ngay cả những phần mềm thống kê nhiều người biết cũng bị ảnh hưởng bởi những lỗi này, dẫn đến ước lượng sai. Những vấn đề này là rất thường gặp trong hồi quy đa thức, bởi vì bình phương và lập phương một con số lớn có thể trở thành khổng lồ. Chuẩn hoá sẽ giải quyết phần lớn vấn đề này. Nó nên là thao tác mặc định của bạn.

Để định nghĩa mô hình parabol, chỉ cần thay đổi định nghĩa $\mu$. Mô hình như sau:

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_1 x_i + \beta_2 x_i^2 \\
\alpha &\sim \text{Normal}(178, 20) \\
\beta_1 &\sim \text{Log-Normal} (0, 1) \\
\beta_2 &\sim \text{Normal} (0, 1) \\
\sigma &\sim \text{Uniform} (0, 50) \\
\end{aligned}$$

Cái khó ở đây là gán prior cho $\beta_2$, tham số cho giá trị bình phương của $x$. Không giống như $\beta_1$, ta không cần ràng buộc nó số dương. Ở phần bài tập cuối chương, bạn sẽ dùng mô phỏng dự đoán prior để hiểu tại sao. Đa số các tham số đa thức rất khó hiểu, nhưng mô phỏng dự đoán prior sẽ giúp rất nhiều.

Ước lượng posterior thì khá dễ. Chỉ cần thay đổi định nghĩa của `mu` để nó chứa cả số hạng tuyến tính và số hạng bậc 2. Nhưng thông thường thì chúng ta nên tiền xử lý chuyển đối biến số - bạn không muốn máy tính thực hiện lại thao tác chuyển đổi cho mỗi lượt của quy trình fit. Cho nên tôi sẽ tính bình phương là `weight_s` như là một biến khác biệt:

<b>code 4.65</b>
```python
d["weight_s"] = (d.weight - d.weight.mean()) / d.weight.std()
d["weight_s2"] = d.weight_s ** 2
def model(weight_s, weight_s2, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", a + b1 * weight_s + b2 * weight_s2)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
m4_5 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m4_5,
    optim.Adam(0.3),
    Trace_ELBO(),
    weight_s=d.weight_s.values,
    weight_s2=d.weight_s2.values,
    height=d.height.values,
)
p4_5, losses = svi.run(random.PRNGKey(0), 3000)
```

Bây giờ, bởi vì quan hệ giữa kết cục `height` và biến dự đoán `weight` phụ thuộc vào 2 slope, `b1` và `b2`, nó không dễ đến đọc mối quan hệ dựa vào bảng các hệ số:

<b>code 4.66</b>
```python
samples = m4_5.sample_posterior(random.PRNGKey(1), p4_5, (1000,))
print_summary({k: v for k, v in samples.items() if k != "mu"}, 0.89, False)
```
<samp>           mean       std    median      5.5%     94.5%     n_eff     r_hat
    a    146.05      0.36    146.03    145.47    146.58   1049.96      1.00
   b1     21.75      0.30     21.75     21.25     22.18    886.88      1.00
   b2     -7.79      0.28     -7.79     -8.21     -7.32   1083.62      1.00
sigma      5.78      0.17      5.78      5.49      6.02    973.21      1.00</samp>

Tham số $\alpha$ (`a`) vẫn là intercept, cho nên nó cho ta biết giá trị `height` mong đợi khi `weight` ở giá trị trung bình. Nhưng nó không còn bằng với trung bình của chiều cao trong mẫu, bởi vì nó không đảm bảo khi ở trong hồi quy đa thức.<sup><a name="r76" href="#76">76</a></sup> Và những tham số $\beta_1$ và $\beta_2$ là thành phần tuyến tính và bậc hai của đường cong. Nhưng cũng không giúp nó rõ ràng hơn.

Ta phải vẽ biểu đồ mức độ fit của những mô hình này để hiểu chúng nói gì. Hãy làm chuyện đó. Chúng ta sẽ tính trung bình của mối quan hệ và khoảng 89% của trung bình và dự đoán, như đã thực hiện ở phần trước. Đây là đoạn code là chuyện đó.

<b>code 4.67</b>
```python
weight_seq = jnp.linspace(start=-2.2, stop=2, num=30)
pred_dat = {"weight_s": weight_seq, "weight_s2": weight_seq ** 2}
post = m4_5.sample_posterior(random.PRNGKey(1), p4_5, (1000,))
predictive = Predictive(m4_5.model, post)
mu = predictive(random.PRNGKey(2), **pred_dat)["mu"]
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
sim_height = predictive(random.PRNGKey(3), **pred_dat)["height"]
height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
```

Vẽ biểu đồ thì đơn giản hơn:

<b>code 4.68</b>
```python
az.plot_pair(
    d[["weight_s", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5}
)
plt.plot(weight_seq, mu_mean, "k")
plt.fill_between(weight_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
plt.fill_between(weight_seq, height_PI[0], height_PI[1], color="k", alpha=0.15)
```

<a name="f11"></a>![](/assets/images/fig 4-11.svg)
<details class="fig"><summary>Hình 4.11: Hồi quy đa thức của chiều cao trên trọng lượng (chuẩn hoá) trong toàn bộ data !Kung. Ở mỗi biểu đồ, data thô được thể hiện bằng các điểm. Đường nét liền là con đường của $\mu$ trong mỗi mô hình, và vùng tô màu là khoảng 89% của trung bình (gần đường nét liền) và khoảng 89% của dự đoán (rộng hơn). Trái: Hồi quy tuyến tính. Giữa: Hồi quy đa thức bậc hai, parabol hay bình phương. Phải: Hồi quy đa thức bậc ba, hay lập phương.</summary>
{% highlight python %}
d["weight_s"] = (d.weight - d.weight.mean()) / d.weight.std()
d["weight_s2"] = d.weight_s ** 2
d["weight_s3"] = d.weight_s ** 3
def model1(weight_s, weight_s2=None, weight_s3=None, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic('mu', a + b1 * weight_s)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height) 
def model2(weight_s, weight_s2=None, weight_s3=None, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic('mu', a + b1 * weight_s + b2 * weight_s2)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)     
def model3(weight_s, weight_s2=None, weight_s3=None, height=None):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    b3 = numpyro.sample("b3", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic('mu', a + b1 * weight_s + b2 * weight_s2 + b3 * weight_s3)
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
weight_seq = jnp.linspace(start=-2.2, stop=2, num=30)
pred_dat = {"weight_s" :weight_seq,
            "weight_s2":weight_seq**2,
            "weight_s3":weight_seq**3,}
fig, axs = plt.subplots(1,3,figsize=(12,5))
for idx, md in enumerate([model1, model2, model3]):
    axs[idx].scatter(d['weight_s'], d['height'])
    guide = AutoLaplaceApproximation(md)
    svi = SVI(
        md,
        guide,
        optim.Adam(0.3),
        Trace_ELBO(),
        weight_s=d.weight_s.values,
        weight_s2=d.weight_s2.values,
        weight_s3=d.weight_s3.values,
        height=d.height.values,
    )
    param, losses = svi.run(random.PRNGKey(0), 3000)
    post = guide.sample_posterior(random.PRNGKey(1), param, (1000,))
    predictive = Predictive(md, post)
    mu = predictive(random.PRNGKey(2), **pred_dat)["mu"]
    mu_mean = jnp.mean(mu, 0)
    mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
    sim_height = predictive(random.PRNGKey(3), **pred_dat)["height"]
    height_PI = jnp.percentile(sim_height, q=(5.5, 94.5), axis=0)
    axs[idx].plot(weight_seq, mu_mean, "C1", linewidth=1)
    axs[idx].fill_between(weight_seq, mu_PI[0], mu_PI[1], color="C2", alpha=1)
    axs[idx].fill_between(weight_seq, height_PI[0], height_PI[1], color="C2", alpha=0.3)
for idx, n in enumerate(['tuyến tính', 'bậc hai', 'bậc 3']):
    axs[idx].set(title=n, xlabel='trọng lượng được chuẩn hoá', ylabel='chiều cao')
{% endhighlight %}
</details>

Kết quả được hiện trong [**HÌNH 4.11**](#f11). Bên trái của hình là hồi quy tuyến tính quen thuộc từ phần trước, nhưng bây giờ thì với biến dự đoán được chuẩn hoá và toàn bộ data người lớn và không phải người lớn. Mô hình tuyến tính cho dự đoán tệ rõ, cả ở trọng lượng rất thấp hoặc rất cao. So sánh nó với mô hình ở giữa, la mô hình đa thức bậc hai mới. Đường cong đã làm công việc tốt hơn khi tìm con đường trung tâm qua data.

Bên phải của [**HÌNH 4.11**](#f11) là hồi quy đa thức bậc cao hơn, hồi quy lập phương trên trọng lượng. Mô hình này là:

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3\\
\alpha &\sim \text{Normal}(178, 20) \\
\beta_1 &\sim \text{Log-Normal} (0, 1) \\
\beta_2 &\sim \text{Normal} (0, 1) \\
\beta_3 &\sim \text{Normal} (0, 1) \\
\sigma &\sim \text{Uniform} (0, 50) \\
\end{aligned}$$

Fit mô hình với một ít tuỳ biến trên đoạn mã của mô hình parabol:

<b>code 4.69</b>
```python
d["weight_s3"] = d.weight_s ** 3
def model(weight_s, weight_s2, weight_s3, height):
    a = numpyro.sample("a", dist.Normal(178, 20))
    b1 = numpyro.sample("b1", dist.LogNormal(0, 1))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))
    b3 = numpyro.sample("b3", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a + b1 * weight_s + b2 * weight_s2 + b3 * weight_s3
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
m4_6 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m4_6,
    optim.Adam(0.3),
    Trace_ELBO(),
    weight_s=d.weight_s.values,
    weight_s2=d.weight_s2.values,
    weight_s3=d.weight_s3.values,
    height=d.height.values,
)
p4_6, losses = svi.run(random.PRNGKey(0), 1000)
```

Tính đường cong và khoảng này thì cũng có một ít tuỳ biến với code trước. Đường cong bậc ba này thì linh hoạt hơn parabol, cho nên nó fit data tốt hơn nữa.

Nhưng nó không rõ là những mô hình này được diễn giải như thế nào. Chúng là những mô tả có tính địa tâm tốt. Một, mô hình fit tốt hơn chưa chắc là mô hình tốt hơn. Nó là chủ đề của Chương 7. Hai, mô hình không chứa ý nghĩa sinh học. Chúng ta không học được quan hệ nhân quả nào giữa chiều cao và trọng lượng. Chúng ta sẽ giải quyết vấn đề thứ hai này trễ hơn, trong Chương 16.

<div class="alert alert-info">
<p><strong>Tuyến tính, phép cộng, khiếp.</strong> Mô hình parabol cho $\mu_i$ trên cũng là mô hình "tuyến tính" của trung bình, mặc dù phương trình rõ ràng không phải đường thẳng. Không may, từ "tuyến tính" còn có nhiều ý nghĩa khác nhau trong ngữ cảnh khác nhau, và nhiều người dùng nó khác nhau trong cùng một ngữ cảnh. Chữ "tuyến tính" trong ngữ cảnh này là $\mu_i$ là <i>hàm tuyến tính</i> của bất kỳ một tham số. Mô hình như vậy có ưu thế là fit data vào dễ hơn. Và chúng cũng dễ diễn giải hơn, bởi vì chúng giả định tham số hành động độc lập đến trung bình. Chúng có nhược điểm là bị sử dụng vô tội vạ. Khi bạn có kiến thức chuyên môn, nó sẽ dễ làm tốt hơn một mô hình tuyến tính. Những mô hình này là những thiết bị địa tâm để mô tả tương quan từng phần. Chúng ta nên cảm thấy xấu hổ khi dùng chúng, để ta không trở nên tự mãn với cách giải thích hiện tượng mà chúng trả về.</p></div>

<div class="alert alert-dark">
<p><strong>Trở về thang tự nhiên.</strong> Biểu đồ trong <a href="#f11"><strong>HÌNH 4.11</strong></a> có đơn vị đã chuẩn hoá ở trục hoành. Những đơn vị đó thường gọi là <i>z-scores</i>. Nhưng giả sử bạn fit mô hình bằng biến số đã chuẩn hoá, những muốn vẽ biểu đồ ước lượng theo thang gốc. Tất cả bạn cần là đầu tiên tắt trục hoành ở biểu đồ chỉ bạn vẽ data thô:</p>
<b>code 4.70</b>
{% highlight python %}
ax = az.plot_pair(
    d[["weight_s", "height"]].to_dict(orient="list"), scatter_kwargs={"alpha": 0.5}
)
ax.set(xlabel="weight", ylabel="height", xticks=[])
fig = plt.gcf()
{% endhighlight %}
<p>Đối số <code>xticks=[]</code> ở cuối sẽ tắt trục hoành. Sau đó bạn tạo lại trục hoành bằng lệnh:</p>
<b>code 4.71</b>
{% highlight python %}
at = jnp.array([-2, -1, 0, 1, 2])
labels = at * d.weight.std() + d.weight.mean()
ax.set_xticks(at)
ax.set_xticklabels([round(label, 1) for label in labels])
fig
{% endhighlight %}
<p>Dòng đầu tiên trên định nghĩa các vị trí của nhãn, theo đơn vị chuẩn hoá. Dòng thứ hai lấy những đơn vị đó và chuyển chúng về thang ban đầu. Dòng còn lại sẽ vẽ các vị trí đó (ticks).</p>
</div>

### 4.5.2 Splines

Cách thứ hai để tạo đường cong là xây dựng một thứ gọi là **SPLINE**. Từ *spline* có nguồn gốc là một mảnh gỗ hoặc kim loại dài, mỏng dùng để làm neo ở vài chỗ để giúp người vẽ hoặc nhà thiết kế vẽ các đường cong. Trong thống kê, spline là một hàm làm mượt dựa trên nhiều hàm số thành phần nhỏ hơn. Thực ra có rất nhiều loại Spline. **B-SPLINE** mà chúng ta học là thường gặp nhất. Chữ "B" là "cơ sở (basis)", nghĩa là "thành phần". B-spline tạo ra những hàm số lắc lư (wiggly), gồm nhiều thành phần ít lắc lư đơn giản hơn. Những thành phần này gọi là hàm basis. Mặc dù có nhiều spline hoa mỹ hơn, chúng ta muốn bắt đầu bằng B-spline bởi vì chúng bắt chúng ta phải quyết định vài chọn lựa mà những spline khác tự động hoá. Bạn cần phải hiểu B-spline trước khi hiểu những spline hoa mỹ hơn.

Để bắt đầu xem cách hoạt động của B-spline, chúng ta cần một ví dụ lắc lư nhiều - một từ khoa học - hơn là data con người !Kung. Hoa anh đào nở hoa khắp nước Nhật vào mùa xuân hàng năm, và do đó có truyền thống xem hoa (*Hanami* 花見). Thời điểm để nở hoa có thể dao động rất nhiều tuỳ theo năm và thập kỷ. Ta hãy tải những ngày nởhoa trong một ngàn năm:

<b>code 4.72</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/cherry_blossoms.csv?raw=true", sep=";")
d.info()
d.describe()
```
<samp>&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 1215 entries, 0 to 1214
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   year        1215 non-null   int64  
 1   doy         827 non-null    float64
 2   temp        1124 non-null   float64
 3   temp_upper  1124 non-null   float64
 4   temp_lower  1124 non-null   float64
dtypes: float64(4), int64(1)
memory usage: 47.6 KB
              year         doy         temp   temp_upper   temp_lower
count  1215.000000  827.000000  1124.000000  1124.000000  1124.000000
mean   1408.000000  104.540508     6.141886     7.185151     5.098941
std     350.884596    6.407036     0.663648     0.992921     0.850350
min     801.000000   86.000000     4.670000     5.450000     0.750000
25%    1104.500000  100.000000     5.700000     6.480000     4.610000
50%    1408.000000  105.000000     6.100000     7.040000     5.145000
75%    1711.500000  109.000000     6.530000     7.720000     5.542500
max    2015.000000  124.000000     8.300000    12.100000     7.740000</samp>

Chúng ta sẽ thực hành trên ghi nhận ngày đầu tiên nở hoa, `doy`. Nó có khoảng từ 86 (cuối tháng 3) đến 124 (đầu tháng 5). Năm ghi nhận hoa nở kéo dài từ 801 CE đến 2015 CE. Bạn nên tiếp tục và vẽ biểu đồ quan hệ giữa `doy` và `year`. Có thể có vài xu hướng lắc lư trong đám mây đó. Rất khó để thấy ra.

Chúng ta hãy thử tách xu hướng đó ra bằng B-spline. Lời giải thích nhanh cho B-spline là nó chia toàn bộ khoảng giới hạn của biến dự đoán, như `year`, thành các bộ phận. Sau đó, chúng gán một tham số cho mỗi thành phần đó. Những tham số này được dần dần mở và tắt để tổng của chúng thành một đường cong hoa mỹ, lắc lư. Mục tiêu cuối cùng là tạo đường zigzag từ những hàm số ít zigzag hơn. Lời giải thích dài thì chứa nhiều chi tiết hơn. Nhưng tất cả những chi tiết đó tồn tại chỉ để đạt được mục tiêu xây dụng một hàm lớn, cong từ nhiều hàm cục bỏ riêng lẻ ít cong hơn.

Đây là lời giải thích dài, với hình ảnh minh hoạ. Mục tiêu của ta là ước lượng xu hướng nở hoa bằng một hàm lắc lư. Với B-spline, cũng giống như hồi quy đa thức, chúng ta tạo ra biến dự đoán mới và sử dụng chúng trong mô hình tuyến tính, $\mu_i$. Khác với hồi quy đa thức, B-spline không chuyển đổi trực tiếp biến dự đoán bằng bình phương hay lập phương. Mà nó tự tổng hợp ra một dãy biến dự đoán mới hoàn toàn. Mỗi biến được tổng hợp này tồn tại chỉ để mở hoặc tắt một tham số cụ thể trong khoảng cụ thể của biến dự đoán thực tế. Mỗi biến được tổng hợp này được gọi là **HÀM CƠ SỞ (BASIS FUNCTION)**. Mô hình tuyến tính sẽ trông rất quen thuộc:

$$\mu_i = \alpha + w_1B_{i,1} + w_2B_{i,2} + w_3B_{i,3} + ... $$

Trong đó, $B_{i,n}$ là hàm cơ sở thứ *n* tại dòng $i$, và các tham số $w$ tương ứng trọng số cho mỗi hàm cơ bản đó. Những tham số $w$ này giống như slope, tuỳ chỉnh ảnh hưởng của mỗi hàm cơ sở lên trung bình $\mu_i$. Cho nên đây cũng là một hồi quy tuyến tính khác, nhưng với những biến dự đoán hoa mỹ được tổng hợp. Những biến được tổng hợp này sẽ làm giúp hoàn thành tốt những công việc mô tả (địa tâm) cho chúng ta.

<a name="f12"></a>![](/assets/images/fig 4-12.svg)
<details class="fig"><summary>Hình 4.12: Sử dụng B-spline để tạo ước lượng khu trú, tuyến tính. Trên: Mỗi hàm cơ sở là một biến số tắt mở ở khoảng cụ thể cho biến dự đoán. Với mỗi giá trị cụ thể trên trục hoành, ví dụ như 1200, chỉ có hai hàm có giá trị không phải zero. Giữa: Tham số được gọi là trọng số nhân với hàm cơ sở. Spline tại một điểm cụ thể là tổng của tất cả những hàm cơ sở được đặt trọng số. Dưới: Kết quả B-spline được thể hiện đối với data. Mỗi tham số trọng số quyết định slope trong khoảng cụ thể của biến dự đoán.</summary>
{% highlight python %}
num_knots = 5
degree = 1
knots = d2['year'].quantile(np.linspace(0,1,num_knots)).to_list()
knots = np.pad(knots, (degree, degree), mode="edge")
B = BSpline(knots, np.identity(num_knots+degree-1), k=degree)(d2['year'])
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + B @ w)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
start = {"w": jnp.zeros(B.shape[1])}
m4_7 = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(model, m4_7, optim.Adam(1), Trace_ELBO(), B=B, D=d2.doy.values)
p4_7, losses = svi.run(random.PRNGKey(0), 20000)
post = m4_7.sample_posterior(random.PRNGKey(1), p4_7, (1000,))
w = jnp.mean(post["w"], 0)
y1200_idx = np.where(d2['year']==1200)
y1200_p = B[y1200_idx][B[y1200_idx].nonzero()]
mu = post["mu"]
mu_PI = jnp.percentile(mu, q=(1.5, 98.5), axis=0)
fig, axs = plt.subplots(3,1,figsize=(12,15))
for i in range(B.shape[1]):
    axs[0].plot(d2.year, B[:, i], "k", alpha=0.5)
axs[0].scatter(knots[degree:-degree], np.repeat(1.05, num_knots), marker="+")
for i in range(num_knots):
    axs[0].annotate(i, (knots[degree:-degree][i]-10, 0.9),color="C0")
axs[0].axvline(1200, 0,1, linestyle="dashed")
axs[0].annotate("1200", (1170, 1.05))
axs[0].scatter(np.repeat(1200,len(y1200_p)), y1200_p,s=100, alpha=1)
axs[0].set(xlabel='năm', ylabel="basis")
for i in range(B.shape[1]):
    axs[1].plot(d2.year, (w[i] * B[:, i]), "k", alpha=0.5)
axs[1].axvline(1200, 0,1, linestyle="dashed")
axs[1].scatter(knots[degree:-degree], np.repeat(5, num_knots), marker="+")
axs[1].set(xlabel='năm', ylabel="basis * weight")
axs[2].scatter(d2.year, d2.doy)
axs[2].fill_between(d2.year, mu_PI[0], mu_PI[1], color="k", alpha=0.5)
axs[2].set(xlabel='năm', ylabel="ngày nở hoa")
plt.tight_layout()
{% endhighlight %}
</details>

Nhưng ta làm thế nào để tạo những biến cơ sở $B$? Tôi thể hiện trường hợp đơn giản nhất trong [**HÌNH 4.12**](#f12), trong đó tôi ước lượng data nở hoa với kết hợp của nhiều ước lượng tuyến tính. Đầu tiên, tôi chia toàn bộ khoảng giới hạn của data thành 4 thành phần, sử dụng điểm mốc gọi là **KNOT**. Những mốc này thể hiện bằng dấu "+" ở hình trên cùng. Tôi đặt những mốc này ở những bách phân vị chẵn của data nở hoa. Trong data nở hoa này, có ít hơn những ghi nhận trong quá khứ xa xôi. Cho nên sử dụng những bách phân vị chẵn không tạo ra những cột mốc cách đều nhau. Đó là lý do tại sao mốc thứ hai rất xa mốc thứ nhất. Bây giờ đừng để ý đến đoạn code để tạo ra những điểm mốc này. Bạn sẽ thấy chúng sau.

Hãy tập trung vào toàn cảnh. Những điểm này là mốc cho 5 hàm cơ sở khác nhau, là các biến $B$. Những biến được tổng hợp này dùng để chuyển đổi nhẹ nhàng từ vùng trước ở trục hoành sang vùng tiếp theo. Chính xác hơn, những biến này cho bạn biết đang gần với điểm mốc nào. Từ bên trái của biểu đồ trên cùng, hàm basis 1 có giá trị là 1, tất cả hàm còn lại là zero. Khi di chuyển dần ra bên phải đến mốc thứ hai thì basis 1 giảm dần và basis 2 tăng dần. Tại mốc 2, basis 2 có giá trị 1, basis khác là zero.

Một đặc trưng hay của những hàm basis này là nó ảnh hưởng khu trú lên các tham số. Tại một điểm bất kỳ ở trục hoành trong [**HÌNH 4.12**](#f12), chỉ hai hàm basis có giá trị non-zero. Ví dụ, đường nét đứt trong biểu đồ trên cùng hiển thị năm 1200. Hàm Basis 1 và 2 là không phải zero vào năm đó. Vậy các tham số cho hàm basis 1 và 2 chỉ là tham số duy nhất ảnh hưởng dự đoán vào năm 1200. Nó khác với hồi quy đa thức, khi tham số ảnh hưởng toàn bộ hình dáng đường cong.

Trong biểu đồ ở giữa [**HÌNH 4.12**](#f12), tôi thể hiện mỗi hàm basis nhân với tham số trọng số tương ứng. Những trọng số này là từ kết quả của việc fit data vào model. Tôi sẽ hướng dẫn sau. Hãy tập trung vào biểu đồ. Những tham số trọng số này có thể dương hoặc âm. Ví dụ như hàm basis 5 xuống thấp hơn dưới zero. Nó có trọng số âm. Để tạo dự đoán cho một năm bất kỳ, ví dụ như năm 1200 lần nữa, chúng ta chỉ cần cộng tất cả những hàm basis đã được nhân với trọng số ở năm đó. Tổng của chúng hơi cao hơn zero (trung bình).

Cuối cùng, biểu đồ ở dưới của [**HÌNH 4.12**](#f12), tôi vẽ các đường spline, tương ứng với khoảng tin cậy 97% của $\mu$, trên toàn bộ data thô. Hầu như tất cả đều bắt được sự thay đổi trong xu hướng quanh 1800. Bạn có thể đoán điều này phản ảnh cho thời tiết toàn cầu như thế nào. Nhưng có nhiều thứ còn trong data, trước năm 1800. Để nhìn thấy rõ hơn, chúng ta làm 2 thứ. Hoặc dùng nhiều điểm mốc hơn, càng nhiều thì spline càng linh hoạt. Hoặc thay vì dùng ước lượng linear, ta có thể dùng đa thức bậc cao.

Bây giờ chúng ta sẽ dùng code để vẽ lại biểu đồ trong [**HÌNH 4.12**](#f12), những cũng cho phép ta thay đổi số lượng các điểm mốc và bậc đa thức mà bạn muốn. Đầu tiên, chúng ta chọn các điểm mốc. Nhớ lại rằng, điểm mốc chính chỉ là các giá trị của `year` đóng vai trờ trạm trung chuyển của spline. Những điểm mốc nên để ở đâu? Có nhiều cách để trả lời câu hỏi này.<sup><a name="r77" href="#77">77</a></sup> Bạn có thể, theo nguyên tắc, đặt bất kỳ chỗ nào. Vị trí của chúng là một phần của mô hình, và bạn có trách nhiệm với chúng. Hãy làm theo những gì chúng đã nói trong ví dụ đơn giản trên, để các điểm mốc ở bách phân vị đều chẵn đều nhau của biến dự đoán. Nó cho bạn nhiều điểm mốc hơn ở nơi nào có nhiều mẫu quan sát. Chúng ta đã dùng chỉ 5 điểm mốc ở ví dụ đầu tiên. Bây giờ ta hãy dùng 15 điểm mốc:

<b>code 4.73</b>
```python
d2 = d[d.doy.notna()]  # complete cases on doy
num_knots = 15
knot_list = jnp.quantile(
    d2.year.values.astype(float), q=jnp.linspace(0, 1, num=num_knots)
)
```

Bạn có thể kiểm tra `knot_list` để xem chúng có đủ 15 giá trị.

Lựa chọn tiếp theo là chọn bậc đa thức. Nó quyết định cách mà các hàm basis gộp lại, từ đó quyết định sự tương tác giữa các tham số để tạo spline. Với bậc 1, như [**HÌNH 4.12**](#f12), hai hàm cộng lại tại mỗi điểm. Với bậc 2, có 3 hàm cộng lại tại mỗi điểm. Bậc 3 có 4 điểm cộng lại. Trong package `scipy` (`jax` chưa có), có hàm tạo ra các hàm basis cho bất kỳ danh sách điểm mốc và bậc đa thức. Đoạn code này sẽ tạo ra các hàm basis cần thiết cho spline bậc 3.

<b>code 4.74</b>
```python
knots = jnp.pad(knot_list, (3, 3), mode="edge")
B = BSpline(knots, jnp.identity(num_knots + 2), k=3)(d2.year.values)
```

Ma trận `B` có 827 dòng và 17 cột. Mỗi dòng là một năm, tương ứng với các dòng trong DataFrame `d2`. Mỗi cột là một hàm basis, một trong những biến số được tổng hợp được định nghĩa trong một khoảng thời gian tương ứng với tham số sẽ ảnh hưởng dự đoán. Để hiện thị hàm basis, chỉ cần vẽ mỗi cột đối với năm:

<b>code 4.75</b>
```python
plt.subplot(
    xlim=(d2.year.min(), d2.year.max()),
    ylim=(0, 1),
    xlabel="year",
    ylabel="basis value",
)
for i in range(B.shape[1]):
    plt.plot(d2.year, B[:, i], "k", alpha=0.5)
```

Tôi hiển thị những hàm basis bậc 3 này ở biểu đồ trên của [**HÌNH 4.13**](#f13).

Để có được tham số trọng số cho mỗi hàm basis, chúng ta cần phải định nghĩa mô hình và cho nó chạy. Mô hình cần thiết chỉ là hồi quy tuyến tính. Hàm basis được tổng hợp sẽ là tất cả công việc. Chúng ta sẽ sử dụng mỗi cột trong ma trận `B` như một biến số. Chúng ta cũng sẽ có một intercept để bắt được ngày nở hoa trung bình. Điều này cũng giúp chúng ta dễ hơn trong việc định nghĩa prior cho trọng số của các basis, bởi vì sau đó chúng ta có thể tưởng tượng chúng như độ lệch chuẩn từ intercept.

Ở dạng toán học, chúng ta bắt đầu bằng xác suất của data và mô hình tuyến tính:

$$ \begin{aligned}
D_i &\sim \text{Normal}(\mu_i. \sigma) \\
\mu_i &= \alpha + \sum_{k=1}^{K} w_k B_{k,i}\\
\end{aligned}$$

Và các prior:

$$\begin{aligned}
\alpha &\sim \text{Normal} (100,10)\\
w_j &\sim \text{Normal} (0,10)\\
\sigma &\sim \text{Exponential}(1)\\
\end{aligned}$$

<a name="f13"></a>![](/assets/images/fig 4-13.svg)
<details class="fig"><summary>Hình 4.13: Spline bậc ba với 15 điểm mốc. Biểu đồ trên cùng, cũng giống trong hình trước, là các hàm basis. Tuy nhiên bây giờ chúng chồng lắp nhiều hơn. Biểu đồ ở giữa là lần nữa các basis được nhân với trọng số tương ứng của chúng. Và tổng của những hàm basis được đặt trọng số này, tại mỗi điểm, tạo ra spline ở biểu đồ dưới, thể hiện khoảng posterior 97% của $\mu$.</summary>
{% highlight python %}
num_knots = 15
degree = 3
knots = d2['year'].quantile(np.linspace(0,1,num_knots)).to_list()
knots = np.pad(knots, (degree, degree), mode="edge")
B = BSpline(knots, np.identity(num_knots+degree-1), k=degree)(d2['year'])
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + B @ w)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
start = {"w": jnp.zeros(B.shape[1])}
m4_7 = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(model, m4_7, optim.Adam(1), Trace_ELBO(), B=B, D=d2.doy.values)
p4_7, losses = svi.run(random.PRNGKey(0), 20000)
post = m4_7.sample_posterior(random.PRNGKey(1), p4_7, (1000,))
w = jnp.mean(post["w"], 0)
y1200_idx = np.where(d2['year']==1200)
y1200_p = B[y1200_idx][B[y1200_idx].nonzero()]
mu = post["mu"]
mu_PI = jnp.percentile(mu, q=(1.5, 98.5), axis=0)
fig, axs = plt.subplots(3,1,figsize=(12,15))
for i in range(B.shape[1]):
    axs[0].plot(d2.year, B[:, i], "k", alpha=0.5)
axs[0].scatter(knots[degree:-degree], np.repeat(1.05, num_knots), marker="+")
for i in range(num_knots):
    axs[0].annotate(i, (knots[degree:-degree][i]-10, 0.9),color="C0")
axs[0].axvline(1200, 0,1, linestyle="dashed")
axs[0].annotate("1200", (1170, 1.05))
axs[0].scatter(np.repeat(1200,len(y1200_p)), y1200_p,s=100, alpha=1)
axs[0].set(xlabel='năm', ylabel="basis")
for i in range(B.shape[1]):
    axs[1].plot(d2.year, (w[i] * B[:, i]), "k", alpha=0.5)
axs[1].axvline(1200, 0,1, linestyle="dashed")
axs[1].scatter(knots[degree:-degree], np.repeat(5, num_knots), marker="+")
axs[1].set(xlabel='năm', ylabel="basis * weight")
axs[2].scatter(d2.year, d2.doy)
axs[2].fill_between(d2.year, mu_PI[0], mu_PI[1], color="k", alpha=0.5)
axs[2].set(xlabel='năm', ylabel="ngày nở hoa")
plt.tight_layout()
{% endhighlight %}
</details>

Mô hình này có vẻ lạ, nhưng những gì nó làm là nhân mỗi giá trị basis với tham số $w_k$ tương ứng, và sau đó lấy tổng tất cả $K$ giá trị tích. Đây chỉ là một phương pháp gọn nhẹ để viết mô hình tuyến tính. Còn lại thì quen thuộc hơn. Mặc dù tôi sẽ yêu cầu bạn mô phỏng những prior đó trong bài tập cuối chương. Bạn có thể đã đoán ra rằng những prior của $w$ ảnh hưởng như thế nào đến độ lắc lư của spline.

Đây cũng là lần đầu tiên chúng ta dùng **PHÂN PHỐI EXPONENTIAL** làm prior. Phân phối exponential rất có ích khi làm prior cho tham số scale, những tham số cần phải dương. Prior của $\sigma$ có phân phối exponential rate là 1. Một cách để đọc phân phối exponential là nó chứa thông tin không gì khác ngoài độ lệch trung bình. Trung bình đó là đảo ngược của rate. Vậy trong trường hợp này là nó là $1/1 = 1$. Nếu rate là 0.5, thì trung bình sẽ là $1/0.5=2$. Chúng ta sẽ dùng phân phối prior là exponential rất nhiều trong sách này, thay thế cho prior uniform. Thông thường thì bạn nên tập trung vào độ lệch trung bình hơn là cực đại.

Để fit mô hình, chúng ta cần một cách để tính tổng đó. Đơn giản nhất là dùng phép nhân ma trận. Nếu bạn không quen với đại số tuyến tính trong bối cảnh này, không sao. Trong phần thông tin cuối bài sẽ có nhiều chi tiết hơn tại sao nó hoạt động được. Mánh duy nhất ở đây là dùng danh sách bắt đầu các trọng số để nói cho `SVI` biết có bao nhiêu cái.

<b>code 4.76</b>
```python
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + B @ w)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
start = {"w": jnp.zeros(B.shape[1])}
m4_7 = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(model, m4_7, optim.Adam(1), Trace_ELBO(), B=B, D=d2.doy.values)
p4_7, losses = svi.run(random.PRNGKey(0), 20000)
```

Bạn có thể nhìn vào trung bình posterior nếu bạn thích với hàm `print_summary`. Những nó sẽ không cho biết nhiều, Bạn sẽ thấy có đến 17 tham số `w`. Nhưng bạn không rõ mô hình nghĩ gì từ những tóm tắt tham số này. Thay vào đó, chúng ta cần vẽ biểu đồ dự đoán posterior. Đầu tiên đây là các hàm basis sau khi được nhân với trọng số:

<b>code 4.77</b>
```python
post = m4_7.sample_posterior(random.PRNGKey(1), p4_7, (1000,))
w = jnp.mean(post["w"], 0)
plt.subplot(
    xlim=(d2.year.min(), d2.year.max()),
    ylim=(-6, 6),
    xlabel="year",
    ylabel="basis * weight",
)
for i in range(B.shape[1]):
    plt.plot(d2.year, (w[i] * B[:, i]), "k", alpha=0.5)
```

Biểu đồ này, với các điểm mốc để tham khảo, được thể hiện ở giữa của [**HÌNH 4.13**](#f13). Và cuối cùng khoảng 97% posterior cho $\mu$, tại mỗi năm:

<b>code 4.88</b>
```python
mu = post["mu"]
mu_PI = jnp.percentile(mu, q=(1.5, 98.5), axis=0)
az.plot_pair(
    d2[["year", "doy"]].astype(float).to_dict(orient="list"),
    scatter_kwargs={"c": "royalblue", "alpha": 0.3, "markersize": 10},
)
plt.fill_between(d2.year, mu_PI[0], mu_PI[1], color="k", alpha=0.5)
```

Nó được thể hiện ở biểu đồ dưới của hình. Spline lúc này thì lắc lư hơn. Có gì đó xảy ra vào năm 1500. Nếu bạn thêm nhiều điểm mốc vào, bạn có thể làm nó càng lắc lư hơn. Bạn tự hỏi bao nhiêu điểm mốc là đủ. Chúng ta sẽ sắn sàng trả lời câu hỏi đó trong vài chương sau. Thực ra chúng ta sẽ trả lời nó bằng cách thay đổi câu hỏi. Vậy nên hãy tạm hoãn lại, và chúng ta sẽ quay lại với nó sau.

Trích xuất xu hướng qua các năm cho chúng ta nhiều thông tin. Nhưng năm không phải là một biến nhân quả, nó chỉ là một bước đệm cho các đặc trưng của từng năm. Trong phần bài tập dưới, bạn sẽ so sánh xu hướng này với các ghi nhận nhiệt độ, để giải thích sự lắc lư.

<div class="alert alert-dark">
<p><strong>Phép nhân ma trận trong mô hình spline.</strong> Đại số ma trận là một chủ đề nhức đầu cho nhiều nhà khoa học. Nếu bạn từng tham gia học nó, thì những gì nó làm là rõ ràng. Nhưng nếu bạn chưa, thì nó là một bí ẩn. Đại số ma trận là một trên gọi mới đại diện cho đại số thông thường. Nó thường gọn hơn. Để tạo mô hình <code>m4_7</code> được lập trình dễ hơn, chúng ta sử dụng phép nhân ma trận của ma trận basis <code>B</code> với vector các tham số <code>w</code>: <code>B @ w</code>. Ký hiệu này chỉ là viết tắt trong đại số tuyến tính cho (1) nhân mỗi thành phần của vector <code>w</code> bởi từng giá trị trong dòng tương ứng của <code>B</code> và sau đó (2) cộng chúng lại. Bạn cũng có thể fit mô hình với code xấu xí hơn:</p>

<b>code 4.79</b>
{% highlight python %}
def model(B, D):
    a = numpyro.sample("a", dist.Normal(100, 10))
    w = numpyro.sample("w", dist.Normal(0, 10).expand(B.shape[1:]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + jnp.sum(B * w, axis=-1))
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
start = {"w": jnp.zeros(B.shape[1])}
m4_7alt = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=start))
svi = SVI(model, m4_7alt, optim.Adam(1), Trace_ELBO(), B=B, D=d2.doy.values)
p4_7alt, losses = svi.run(random.PRNGKey(0), 20000)
{% endhighlight %}
<p>Và bạn có được chính xác những gì bạn cần: Tổng dự đoán tuyến tính cho mỗi năm (dòng). Nếu bạn không làm việc được với đại số tuyến tính, ký hiệu ma trận có thể gây khó khăn. Nó tốt hơn khi nhớ rằng là không gì hơn ngoài toán học bạn đã biết, nhưng ở dạng nén nhỏ, mà rất tiện khi ứng dụn trong các phép tính lặp đi lặp lại trên một dãy số.</p>
</div>

### 4.5.3 Nhưng hàm làm mượt khác của thế giới gồ ghề

Spline ở phần trước chỉ mới bắt đầu. Một lớp các mô hình, gọi là **GENERALIZED ADDICTIVE MODELS (GAMs)**, tập trung vào dự báo một biến kết cục dựa trên hàm làm mượt của vài biến dự đoán. Chủ đề này rất sâu đến nổi đủ cần một sách riêng.<sup><a name="r78" href="#78">78</a></sup>

## <center>4.6 Tổng kết</center><a name="a6"></a>

Chương này giới thiệu hồi quy tuyến tính đơn giản, là một khung quy trình để ước lượng mối liên quan giữa biến dự đoán và biến kết cục. Phân phối Gaussian bao gồm likelihood của mô hình này, bởi vì nó đếm số lượng tương đối số cách kết hợp khác nhau của trung bình và độ lệch chuẩn để tạo ra quan sát. Để fit data vào mô hình, chương này giới thiệu phương pháp quadratic approximation và công cụ `SVI`. Nó cũng giới thiệu quy trình mới để vẽ biểu đồ thể hiện phân phối prior và posterior.

Chương sau sẽ mở rộng những khái niệm này, thông qua giới thiệu mô hình hồi quy với nhiều hơn một biến dự đoán. Kỹ thuật cơ bản từ chương này là nền tảng cho hầu hết các ví dụ trong các chương sắp tới. Cho nên nếu những tài liệu này là còn mới với bạn, thì nó đáng để bạn xem lại chương này, trước khi bước đi tiếp theo.

---

<details><summary>Endnotes</summary>
<ol class="endnotes">
<li><a name="65" href="#r65">65. </a>Leo Breiman, at the start of Chapter 9 of his classic book on probability theory (Breiman, 1968), says “there is really no completely satisfying answer” to the question “why normal?” Many mathematical results remain mysterious, even after we prove them. So if you don’t quite get why the normal distribution is the limiting distribution, you are in good company.</li>
<li><a name="66" href="#r66">66. </a>For the reader hungry for mathematical details, see Frank (2009) for a nicely illustrated explanation of this, using Fourier transforms.</li>
<li><a name="67" href="#r67">67. </a>Technically, the distribution of sums converges to normal only when the original distribution has finite variance. What this means practically is that the magnitude of any newly sampled value cannot be so big as to overwhelm all of the previous values. There are natural phenomena with effectively infinite variance, but we won’t be working with any. Or rather, when we do, I won’t comment on it.</li>
<li><a name="68" href="#r68">68. </a>The most famous non-technical book about this topic is Taleb (2007). This book has had a large impact. There is also a quite large technical literature on the topic. Note that the terms heavy tail and fat tail sometimes have precise technical definitions.</li>
<li><a name="69" href="#r69">69. </a>A very nice essay by Pasquale Cirillo and Nassim Nicholas Taleb, “The Decline of Violent Conflicts: What Do The Data Really Say?,” focuses on this issue.</li>
<li><a name="70" href="#r70">70. </a>Howell (2010) and Howell (2000). See also Lee and DeVore (1976). Much more raw data is available for download from https://tspace.library.utoronto.ca/handle/1807/10395.</li>
<li><a name="71" href="#r71">71. </a>Jaynes (2003), page 21–22. See that book’s index for other mentions in various statistical arguments.</li>
<li><a name="72" href="#r72">72. </a>See Jaynes (1986) for an entertaining example concerning the beer preferences of left-handed kangaroos. There is an updated 1996 version of this paper available online.</li>
<li><a name="73" href="#r73">73. </a>The strategy is the same grid approximation strategy as before (page 39). But now there are two dimensions, and so there is a geometric (literally) increase in bother. The algorithm is mercifully short, however, if not transparent. Think of the code as being six distinct commands. The first two lines of code just establish the range of $\mu$ and $\sigma$ values, respectively, to calculate over, as well as how many points to calculate in-between. The third line of code expands those chosen $\mu$ and $\sigma$ values into a matrix of all of the combinations of $\mu$ and $\sigma$. This matrix is stored in a data frame, <code>post</code>. In the monstrous fourth line of code, shown in expanded form to make it easier to read, the log-likelihood at each combination of $\mu$ and $\sigma$ is computed. This line looks so awful, because we have to be careful here to do everything on the log scale. Otherwise rounding error will quickly make all of the posterior probabilities zero. So what sapply does is pass the unique combination of $\mu$ and $\sigma$ on each row of post to a function that computes the log-likelihood of each observed height, and adds all of these log-likelihoods together (<code>sum</code>). In the fifth line, we multiply the prior by the likelihood to get the product that is proportional to the posterior density. The priors are also on the log scale, and so we add them to the log-likelihood, which is equivalent to multiplying the raw densities by the likelihood. Finally, the obstacle for getting back on the probability scale is that rounding error is always a threat when moving from log-probability to probability. If you use the obvious approach, like <code>jnp.exp(post["prod"])</code>, you’ll get a vector full of zeros, which isn’t very helpful. This is a result of R’s rounding very small probabilities to zero. Remember, in large samples, all unique samples are unlikely. This is why you have to work with log-probability. The code in the box dodges this problem by scaling all of the log-products by the maximum log-product. As a result, the values in <code>post["prob"]</code> are not all zero, but they also aren’t exactly probabilities. Instead they are relative posterior probabilities. But that’s good enough for what we wish to do with these values.</li>
<li><a name="74" href="#r74">74. </a>The most accessible of Galton’s writings on the topic has been reprinted as Galton (1989).</li>
<li><a name="75" href="#r75">75. </a>See Reilly and Zeringue (2004) for an example using predator-prey dynamics. example in Chapter 16. [94] We’ll engage with</li>
<li><a name="76" href="#r76">76. </a>The implied definition of α in a parabolic model is $\alpha = E y_i − \beta_1 E x_i − \beta_2 E x_i^2$ . Now even when the average x i is zero, $E x_i = 0$, the average square will likely not be zero. So $\alpha$ becomes hard to directly interpret again.</li>
<li><a name="77" href="#r77">77. </a>For much more discussion of knot choice, see Fahrmeir et al. (2013) and Wood (2017). A common approach is to use Wood’s knot choice algorithm as implemented by default in the R package <code>mgcv</code>.</li>
<li><a name="78" href="#r78">78. </a>A very popular and comprehensive text is Wood (2017).</li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
<p><strong>4E1.</strong> In the model definition below, which line is the likelihood?</p>
$$\begin{aligned}
y_i &\sim \text{Normal}(\mu , \sigma) \\
\mu &\sim \text{Normal}(0, 10) \\
\sigma &\sim \text{Exponential}(1) \\
\end{aligned}$$
<p><strong>4E2.</strong> In the model definition just above, how many parameters are in the posterior distribution? </p>
<p><strong>4E3.</strong> Using the model definition above, write down the appropriate form of Bayes’ theorem that includes the proper likelihood and priors.</p>
<p><strong>4E4.</strong> In the model definition below, which line is the linear model?</p>
$$\begin{aligned}
y_i &\sim \text{Normal}(\mu, \sigma) \\
\mu_i &= \alpha + \beta x_i \\
\alpha &\sim \text{Normal}(0, 10) \\
\beta &\sim \text{Normal}(0, 1) \\
\sigma &\sim \text{Exponential}(2)\\
\end{aligned}$$
<p><strong>4E5.</strong> In the model definition just above, how many parameters are in the posterior distribution?</p>
<p><strong>4M1.</strong> For the model definition below, simulate observed $y$ values from the prior (not the posterior).</p>
$$\begin{aligned}
y_i &\sim \text{Normal}(\mu , \sigma) \\
\mu &\sim \text{Normal}(0, 10) \\
\sigma &\sim \text{Exponential}(1) \\
\end{aligned}$$
<p><strong>4M2.</strong> Translate the model just above into a <code>numpyro</code> model.</p>
<p><strong>4M3.</strong> Translate the <code>numpyro</code> model formula below into a mathematical model definition.</p>
<pre>def model(x, y):
    a = numpyro.sample('a', dist.Normal(0,10) )
    b = numpyro.sample('b', dist.Uniform(0,1) )
    sigma = numpyro.sample('sigma', dist.Exponential(1) )
    mu = numpyro.deterministic('mu', a + b*x )
    numpyro.sample('y', dist.Normal(mu,sigma), obs=y)</pre>
<p><strong>4m4_</strong> A sample of students is measured for height each year for 3 years. After the third year, you want to fit a linear regression predicting height using year as a predictor. Write down the mathematical model definition for this regression, using any variable names and priors you choose. Be prepared to defend your choice of priors.</p>
<p><strong>4M5.</strong> Now suppose I remind you that every student got taller each year. Does this information lead you to change your choice of priors? How?</p>
<p><strong>4M6.</strong> Now suppose I tell you that the variance among heights for students of the same age is never more than 64cm. How does this lead you to revise your priors?</p>
<p><strong>4M7.</strong> Refit model <code>m4_3</code> from the chapter, but omit the mean weight <code>xbar</code> this time. Compare the new model’s posterior to that of the original model. In particular, look at the covariance among the parameters. What is different? Then compare the posterior predictions of both models.</p>
<p><strong>4M8.</strong> In the chapter, we used 15 knots with the cherry blossom spline. Increase the number of knots and observe what happens to the resulting spline. Then adjust also the width of the prior on the weights—change the standard deviation of the prior and watch what happens. What do you think the combination of knot number and the prior on the weights controls?</p>
<p><strong>4H1.</strong> The weights listed below were recorded in the !Kung census, but heights were not recorded for these individuals. Provide predicted heights and 89% intervals for each of these individuals. That is, fill in the table below, using model-based predictions.</p>
<p><table>
    <thead>
        <tr>
            <th>Individual</th>
            <th>weight</th>
            <th>expected height</th>
            <th>89% interval</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>46.95</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>2</td>
            <td>43.72</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>3</td>
            <td>64.78</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>4</td>
            <td>32.59</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>5</td>
            <td>54.63</td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table></p>
<p><strong>4H2.</strong> Select out all the rows in the <code>Howell1</code> data with ages below 18 years of age. If you do it right, you should end up with a new data frame with 192 rows in it.</p>
<ol type="a">
    <li>Fit a linear regression to these data, using <code>SVI</code> and <code>AutoLaplaceApproximation</code>. Present and interpret the estimates. For every 10 units of increase in weight, how much taller does the model predict a child gets?</li>
    <li>Plot the raw data, with height on the vertical axis and weight on the horizontal axis. Superimpose the MAP regression line and 89% interval for the mean. Also superimpose the 89% interval for predicted heights.</li>
    <li>What aspects of the model fit concern you? Describe the kinds of assumptions you would change, if any, to improve the model. You don’t have to write any new code. Just explain what the model appears to be doing a bad job of, and what you hypothesize would be a better model.</li>
</ol>
<p><strong>4H3.</strong> Suppose a colleague of yours, who works on allometry, glances at the practice problems just above. Your colleague exclaims, “That’s silly. Everyone knows that it’s only the <i>logarithm</i> of body weight that scales with height!” Let’s take your colleague’s advice and see what happens.</p>
<ol type="a">
    <li>Model the relationship between height (cm) and the natural logarithm of weight (log-kg). Use the entire Howell1 data frame, all 544 rows, adults and non-adults. Can you interpret the resulting estimates?</li>
    <li>Begin with this plot: <code>az.plot_pair</code>. Then use samples from the quadratic approximate posterior of the model in (a) to superimpose on the plot: (1) the predicted mean height as a function of weight, (2) the 97% interval for the mean, and (3) the 97% interval for predicted heights.</li>
</ol>
<p><strong>4H4.</strong> Plot the prior predictive distribution for the parabolic polynomial regression model in the chapter. You can modify the code that plots the linear regression prior predictive distribution. Can you modify the prior distributions of $\alpha$, $\beta_1$ , and $\beta_2$ so that the prior predictions stay within the biologically reasonable outcome space? That is to say: Do not try to fit the data by hand. But do try to keep the curves consistent with what you know about height and weight, before seeing these exact data.</p>
<p><strong>4H5.</strong> Return to data <code>cherry_blossoms</code> and model the association between blossom date (<code>doy</code>) and March temperature (<code>temp</code>). Note that there are many missing values in both variables. You may consider a linear model, a polynomial, or a spline on temperature. How well does temperature trend predict the blossom trend?</p>
<p><strong>4H6.</strong> Simulate the prior predictive distribution for the cherry blossom spline in the chapter. Adjust the prior on the weights and observe what happens. What do you think the prior on the weights is doing?</p>
<p><strong>4H8.</strong> The cherry blossom spline in the chapter used an intercept $\alpha$, but technically it doesn’t require one. The first basis functions could substitute for the intercept. Try refitting the cherry blossom spline without the intercept. What else about the model do you need to change to make this work?</p></details>