---
title: "Chapter 3: Sampling the Imaginary"
description: "Chương 3: Lấy mẫu từ tưởng tượng"
---

- [3.1 Lấy mẫu từ posterior tính qua grid approximation](#a1)
- [3.2 Lấy mẫu để mô tả](#a2)
- [3.3 Lấy mẫu để mô phỏng dự đoán](#a3)
- [3.4 Tổng kết](#a4)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from jax import vmap, random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Nhiều chương trình dạy thống kê Bayes bằng tình huống xét nghiệm y khoa. Để lặp lại cấu trúc của những ví dụ thường gặp, giả sử có một xét nghiệm ma cà rồng (vampire) chính xác 95%, có thể mô tả bằng ký hiệu $\Pr(\text{dương} \| \text{ma cà rồng}) = 0.95$ . Đây là một xét nghiệm có độ chính xác cao, hầu như chẩn đoán đúng ma cà rồng thật. Nhưng lỗi cũng có thể xảy ra, dưới dạng dương tính giả. Một phần trăm trường hợp, nó chẩn đoán sai người thường là ma cà rồng, $\Pr(\text{dương} \| \text{người}) = 0.01$. Một thông tin khác là ma cà rồng rất hiếm, chiếm 0.1% dân số, tức là $\Pr(\text{ma cà rồng}) = 0.001$. Giả sử một người nào đó bị xét nghiệm dương tính. Vậy người đó có xác suất bao nhiêu phần trăm là ma cà rồng?

Cách tiếp cận đúng là dùng Bayes' theorem để đảo ngược lại xác suất, để tính $\Pr(\text{ma cà rồng} \| \text{dương tính} $. Phép tính có thể trình bày như sau:

$$ \Pr(\text{ma cà rồng}| \text{dương tính} ) = \frac{\Pr(\text{dương tính} |\text{ma cà rồng}) \Pr(\text{ma cà rồng})} {\Pr(\text{dương tính})} $$

Với $\Pr(\text{dương tính})$ = xác suất trung bình của một kết quả xét nghiệm dương tính.

$$ \begin{aligned}
\Pr(\text{dương tính}) = &\Pr(\text{dương tính} |\text{ma cà rồng}) \Pr(\text{ma cà rồng}) \\
&+ \Pr(\text{dương tính} |\text{người thường})(1 - \Pr(\text{ma cà rồng}) \\
\end{aligned}$$

<b>Code 3.1</b>
```python
Pr_Positive_Vampire = 0.95
Pr_Positive_Mortal = 0.01
Pr_Vampire = 0.001
numerator = Pr_Positive_Vampire * Pr_Vampire
denominator = numerator + Pr_Positive_Mortal * (1 - Pr_Vampire)
Pr_Vampire_Positive = numerator / denominator
Pr_Vampire_Positive
```
<samp>0.08683729</samp>

Đối tượng đó chỉ có xác suất 8.7% là ma cà rồng.

Nhiều người sẽ thấy rằng kết quả có vẻ mâu thuẫn. Nhưng đây là trường hợp chung trong nhiều tình huống xét nghiệm, giống như HIV và DNA,.. Mỗi khi nào trạng thái quan tâm rất hiếm xảy ra, kết quả xét nghiệm dương cũng có thể không đảm bảo chắc chắn là dương tính thật, vì có rất nhiều dương tính giả, mặc dù âm tính thật vẫn được phát hiện đúng.

Tôi không thích những ví dụ như vậy, bởi 2 lý do:
- Trước tiên, không có tính chất "Bayes" nào ở đây cả. Hãy nhớ rằng, suy luận bayes được đặc trưng bởi lý thuyết xác suất tổng phát, không phải bởi cách sử dụng công thức Bayes' theorem. Bởi vì tất cả xác suất tôi cung cấp ở trên tương đồng với tần số của dữ kiện, chứa không phải parameter, nên việc dùng Bayes' theorem là chấp nhận được.
- Thứ 2, và quan trọng hơn tới tác vụ của chúng ta trong chương này, chính những ví dụ như vậy làm cho suy luận Bayes trở nên khó tiếp cận hơn. Rất ít người có thể ghi nhớ số nào để ở đâu, có lẽ do họ không nắm được logic của quy trình. Nó giống như là một công thức từ trên trời rơi xuống. Nếu bạn cảm thấy lẫn lộn, đó là do bạn đang cố gắng hiểu nó.

Tuy nhiên, có một dạng trình bày vấn đề trên giúp bạn cảm thấy dễ hiểu hơn. Giả sử thay vì báo cáo các con số xác suất, tôi sẽ nói rằng:
1. Trong dân số 100,000 người, 100 người là ma cà rồng.
2. Trong 100 ma cà rồng đó, 95 trong số đó là có xét nghiệm dương tính.
3. Trong 99,000 người còn lại, 999 trong số đó là xét nghiệm dương tính.

Bây giờ hãy nói cho tôi biết, nếu chúng tôi kiểm tra tất cả 100,000 người, tỉ lệ của những người xét nghiệm dương tính là ma cà rồng thực sự? Rất nhiều người, mặc dù chắc chắn không phải tất cả, sẽ thấy dạng trình bày này dễ hơn nhiều.<sup><a name="r50" href="#50">50</a></sup> Bây giờ chúng ta có thể cộng số người xét nghiệm dương tính: $95+999=1094$. Trong 1094 xét nghiệm dương tính, 95 trong số họ là ma cà rồng thực sự, và suy ra:

$$ \Pr(\text{ma cà rồng}|\text{dương tính}) = \frac{95}{1094} \approx 0.087 $$

Dạng trình bày thứ hai của vấn đề, sử dụng số đếm thay vì xác suất, thường được gọi là *định dạng tần số (frequency format)* hay *tần số tự nhiên (natural frequency)*. Tại sao định dạng này giúp người ta chọn được cách tiếp cận đúng đắn vấn còn là một câu hỏi mở. Người ta cho rằng bộ não của loài người thích nghi tốt hơn khi nó nhận thông tin dưới dạng mà người nhận được trong môi trường tự nhiên. Trong thế giới tự nhiên, chúng ta chỉ gặp số đếm, không ai từng gặp xác suất. Mọi người đều nhìn số đếm hay "tần số" trong cuộc sống mỗi ngày.

Cho dù lời giải thích cho hiện tượng này là gì đi nữa, chúng ta sẽ lợi dụng nó. Trong chương này, chúng ta sẽ áp dụng dạng trình bày này, bằng cách lấy các phân phối xác suất từ chương trước và lấy mẫu từ chúng để tạo các số đếm. Phân phối posterior là một phân phối xác suất. Và cũng giống như các phân phối xác suất khác, ta có thể tưởng tượng lấy mẫu từ nó. Kết quả của việc lấy mẫu này là các giá trị của parameter. Phần lớn các parameter không có thực thể rõ ràng. Thống kê Bayes xem phân phối parameter như các tính phù hợp tương đối, chứ không phải một quy trình xử lý ngẫu nhiên tồn tại trong đời thực. Trong bất kỳ sự kiện nào, sự ngẫu nhiên là một đặc trưng của thông tin, còn ngoài đời thực thì không phải. Nhưng trong máy tính, parameter chỉ là một thực thể giống như con số kết cục của việc tung đồng xu hoặc lắc xí ngầu hoặc một thí nghiệm nông nghiệp. Posterior tạo ra tần suất mong đợi của các giá trị parameter khác nhau mà có thể xuất hiện, khi chúng ta bắt đầu tháo các giá trị parameter từ nó.

<div class="alert alert-info">
	<p><strong>Hiện tượng tần số tự nhiên là không độc nhất.</strong> Thay đổi dạng trình bày của vấn đề thường làm nó nhận biết dễ hơn và khơi dậy những ý tưởng mới mà không thể gặp ở dạng trình bày cũ.<sup><a name="r51" href="#51">51</a></sup> Trong vật lý, thay đổi cơ chế Newtonian và Lagranian có thể giúp giải quyết vấn đề đơn giản hơn. Trong sinh học tiến hoá, thay đổi inclusive fitness và multilevel selection thêm ánh sáng mới vào mô hình cũ. Trong thống kê, chuyển đổi dạng trình bày giữa thống kê Bayes và non-Bayes thường dạy cho ta biết nhiều thứ mới về cả cách tiếp cận.</p>
</div>

Chương này hướng dẫn những kỹ năng cơ bản để làm việc với mẫu lấy từ phân phối. Có vẻ hơi sai khi bây giờ đã làm việc với mẫu, bởi vì posterior trong ví dụ tung quả rất đơn giản. Nó đơn giản đến mức ta có thể tính nó bằng grid approximation hoặc dùng phân tích toán học. Nhưng có hai lý do để ta phải học cách lấy mẫu từ sớm.

Một, nhiều nhà nghiên cứu không quen với tích phân, mặc dù họ biết những phương pháp mạnh và đúng để tóm tắt data. Làm việc với mẫu chuyển câu hỏi tích phân thành câu hỏi tóm tắt data, thành định dạng tần số mà mọi người quen thuộc. Tích phân trong bối cảnh Bayes điển hình sẽ là tổng các xác suất trong khoảng nhất định. Nó có thể là một phép toán đầy thách thức. Nhưng khi ta có mẫu lấy từ phân phối xác suất, nó trở thành là câu hỏi tần số trong một khoảng nhất định. Bằng phương pháp này, nhà nghiên cứu có thể đặt câu hỏi và trả lời nhiều nghi vấn trong mô hình, mà không cần đến chuyên gia toán học.<sup><a name="r52" href="#52">52</a></sup> Vì lý do này, việc lấy mẫu từ phân phối posterior giúp ta có cảm giác trực quan hơn so với làm việc trực tiếp với xác suất và tích phân.

Hai, những phương pháp tìm posterior hiện đại cho kết quả là mẫu chứ không phải phân phối. Đa số các phương pháp này là biến thể của **Markov Chain Monte Carlo (MCMC)**. Cho nên nếu bạn làm quen sớm với các khái niệm và quy trình xử lý mẫu từ phân phối, sau này khi bạn phải bắt buộc fit mô hình bằng MCMC, bạn sẽ biết mình làm được gì với kết quả từ MCMC. Tới Chương 9, bạn sẽ dùng MCMC như chìa khóa mở ra nhiều mô hình đa dạng và phức tạp hơn. MCMC không còn là phương pháp dành riêng cho giới chuyên gia, mà hơn là một công cụ cơ bản cho khoa học định lượng. Cho nên nó đáng để kế hoạch trước.

Vậy trong chương này chúng ta sẽ bắt đầu sử dụng mẫu để tóm tắt và mô phỏng kết quả của mô hình. Kỹ năng bạn học được này sẽ áp dụng cho mọi vấn đề còn lại trong sách, mặc dù chi tiết của mô hình và cách tạo mẫu có thể khác nhau.

<div class="alert alert-info">
	<p><strong>Thống kê không cứu được khoa học kém.</strong> Ví dụ ma cà rồng trên có cấu trúc logic giống như nhiều bài toán <i>phát hiện tín hiệu (signal detection)</i> khác: (1) Có một trạng thái nhị phân bị ẩn đi; (2) ta quan sát được một tín hiệu không hoàn toàn của trạng thái ẩn đó; (3) ta nên dùng Bayes' theorem để suy luận về tác động của tín hiệu đó vào tính bất định cần tìm.</p>
	<p>Suy luận khoa học cũng theo quy trình tương tự: (1) Một giả thuyết có thể đúng hoặc sai; (2) ta dùng tín hiệu thống kê để chứng minh giả thuyết sai; (3) ta nên dùng Bayes' theorem để suy luận tác động của tín hiệu đó vào trạng thái của giả thuyết. Bước thứ ba thường ít được thực hiện, do đó tác giả không thích quy trình này lắm. Nhưng hãy xem ví dụ sau đây, bạn sẽ thấy được ý nghĩa của nó. Giả sử ta có xác suất của một tín hiệu dương tính, khi mà giả thuyết đúng, tức là $\Pr( \text{tín hiệu} \| \text{đúng} ) = 0.95$ . Đây là *sức mạnh (power)* của phép kiểm định. Giả sử ta có thêm xác suất của một tín hiệu dương tính, nhưng với giả thuyết sai, $\Pr( \text{tín hiệu} \| \text{sai} ) = 0.05$ . Đây là xác suất dương tính giả, giống như 5% tiện dụng của các phép kiểm định. Cuối cùng, ta phải có <i>xác suất nền (base rate)</i> khi giả thuyết đúng. Giả sử 1 trong 100 giả thuyết là đúng. Thì $\Pr(đúng) = 0.01$.  Không ai biết giả trị này, nhưng trong lịch sử khoa học thì số này rất nhỏ. Xem Chương 17 để thảo luận thêm. Bây giờ ta tính posterior.</p>
$$\begin{aligned}
\Pr( \text{đúng} | \text{dương} ) &= \frac{ \Pr( \text{dương} | \text{đúng} ) \Pr( \text{đúng} )  }{\ Pr( \text{dương} ) } \\
&= \frac{ \Pr( \text{dương} | \text{đúng} ) \Pr( \text{đúng} ) }{ \Pr( \text{dương} | \text{đúng} ) \Pr( \text{đúng} ) + \Pr( \text{dương} | \text{sai} ) \Pr( \text{sai} ) } \\
\end{aligned}$$  
	<p>Sau khi thay bằng các con số, kết quả là khoảng $ \Pr(\text{đúng}| \text{dương}) = 0.16$. Nghĩa là tín hiệu dương tính chỉ cho 16% xác suất giả thuyết là đúng. Nó giống như hiện tượng xác suất nền nhỏ trong ví dụ xét nghiệm y khoa (và ma cà rồng). Bạn có thể giảm tỉ lệ dương tính giả thấp xuống 1%, và đẩy xác suất posterior lên 0.5, và nó cũng chỉ tốt như việc tung một đồng xu. Quan trọng nhất để làm ở đây là cải thiện xác suất nền, $\Pr(\text{dương})$ và nó cần được suy nghĩ, chứ không phải kiểm định.<sup><a name="r53" href="#53">53</a></sup></p>
</div>

## <center>3.1 Lấy mẫu từ posterior tính qua grid approximation</center><a name="a1"></a>

Trước khi làm việc với mẫu, ta phải tạo được nó. Nhắc lại phương pháp grid approximation để tìm posterior trong ví dụ tung quả cầu. Nên nhớ rằng, *posterior* ở đây nghĩa là xác suất của $p$ đặt điều kiện lên data.

<b>Code 3.2</b>
```python
p_grid = jnp.linspace(start=0, stop=1, num=1000)
prob_p = jnp.repeat(1, 1000)
prob_data = jnp.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6))
posterior = prob_data * prob_p
posterior = posterior / jnp.sum(posterior)
```

Giờ ta muốn lấy 10,000 mẫu từ posterior này. Tưởng tượng posterior như một rổ chứa đầy các giá trị của parameter, các con số như 0.1, 0.7, 0.5, 1,.. Trong cái rổ, mỗi giá trị tồn tại có tần suất tỉ lệ thuận với xác suất posterior, ví dụ như giá trị gần đỉnh thì thường gặp hơn các giá trị ở hai đuôi. Ta lấy 10,000 giá trị từ trong rổ. Cho rằng cái rổ đã trộn đều, các mẫu lấy từ nó sẽ có tỉ lệ thành phần giống hoàn toàn với mật độ posterior. Cho nên mỗi giá trị $p$ sẽ xuất hiện trong mẫu sẽ tỉ lệ với tính phù hợp của mỗi giá trị trong posterior.

Trong python, đây là lệnh để làm việc này.

<b>Code 3.3</b>
```python
samples = p_grid[dist.Categorical(probs=posterior).sample(random.PRNGKey(0), (10000,))]
```

<a name="f1"></a>![](/assets/images/fig 3-1.svg)
<details class="fig"><summary>Hình 3.1: Lấy mẫu các giá trị parameter từ phân phối posterior. Trái: 10,00 mẫu từ posterior suy ra từ data và mô hình tung quả cầu. Phải: mật độ của mẫu (trục tung) ở mỗi giá trị khả dĩ của parameter (trục hoành).</summary>
{% highlight python %}fig, ax = plt.subplots(1, 2, figsize=(10,5))
p_grid = jnp.linspace(0,1,1000)
prob_b = jnp.repeat(1,1000)
prob_data = jnp.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6))
posterior = prob_data * prob_p
posterior = posterior / jnp.sum(posterior)
samples = p_grid[dist.Categorical(probs=posterior).sample(random.PRNGKey(0), (10000,))]
ax[0].scatter(range(len(samples)), samples, alpha=0.15)
ax[0].set(xlabel="số mẫu", ylabel="tỉ lệ nước ($p$)")
az.plot_dist(samples, ax=ax[1])
ax[1].set(xlabel="tỉ lệ nước ($p$)", ylabel="mật độ")
plt.tight_layout(){% endhighlight %}</details>

Động cơ chính ở đây là hàm `sample`của `dist.Categorial`, nó lấy mẫu một cách ngẫu nhiên các giá trị từ một vector. Vector trong trường hợp này là `p_grid`, grid của các giá trị parameter. Xác suất mỗi giá trị được cho trong `posterior`, mà bạn đã tính ở trên.

Kết quả được thể hiện trên [**HÌNH 3.1**](#f1). Nhìn vào biểu đò bên trái, tất cả 10,000 (`1e4`) mẫu ngẫu nhiên được vẽ lên theo tuần tự.

<b>Code 3.4</b>
```python
plt.scatter(range(len(samples)), samples, alpha=0.2)
```

Trong hình này, nó giống như bạn đang bay trên phân phối posterior, và nhìn xuống dưới. Có rất nhiều mẫu từ vùng có mật độ dày ở vùng gần 0.6 và rất ít mẫu dưới 0.25. Ở bên phải, biểu đồ cho thấy *ước lượng mật độ (density estimate)* tính từ các mẫu.

<b>Code 3.5</b>
```python
az.plot_density({"": samples}, hdi_prob=1)
```

Bạn có thể thấy rằng mật độ của ước lượng rất giống với posterior lý tưởng mà ta tính qua grid approximation. Nếu bạn lấy nhiều mẫu hơn, có thể `1e5` hoặc `1e6`, ước lượng mật độ có thể càng ngày càng giống với posterior lý tưởng.

Những gì bạn làm nãy giờ là tái lập lại phân phối posterior mà bạn đã tính qua grid approximation. Bản thân chuyện này thì không có giá trị gì. Nhưng tiếp theo bạn sẽ bắt đầu dùng những mẫu này để mô tả và tìm hiểu posterior. Và đó là một giá trị to lớn.

## <center>3.2 Lấy mẫu để mô tả</center><a name="a2"></a>

Một khi đã tạo được phân phối posterior, nhiệm vụ của mô hình đã xong. Nhưng công việc của bạn chỉ mới bắt đầu. Bạn cần thiết phải phải tóm tắt và diễn giải phân phối posterior. Bằng cách nào thì tuỳ mục đích của bạn. Nhưng câu hỏi thường gặp gồm:

- Xác suất nhỏ hơn một giá trị nào đó là bao nhiêu?
- Xác suất trong khoảng giá trị nào đó là bao nhiêu?
- Giá trị nào dánh dấu 5% dưới của xác suất posterior?
- Khoảng giá trị nào chứa 90% của xác suất posterior?
- Giá trị nào có xác suất cao nhất?

Có thể chia làm 3 nhóm câu hỏi: (1) khoảng *ranh giới xác định (defined boundaries)*; (2) khoảng *mật độ xác suất xác định (defined probability mass)*; (3) *ước lượng điểm (point estimate)*. Chúng ta sẽ thấy làm sao để tiếp cận những câu hỏi này bằng mẫu rút ra từ posterior.

### 3.2.1 Khoảng ranh giới xác định.

Giả sử tôi hỏi bạn xác suất để tỉ lệ bề mặt nước nhỏ hơn 0.5. Bằng posterior từ grid approximation, bạn chỉ việc dùng tổng các xác suất mà giá trị parameter nhỏ hơn 0.5.

<b>Code 3.6</b>
```python
jnp.sum(posterior[p_grid < 0.5])
```
<samp>0.17187457</samp>

Vậy có khoảng 17 % của xác suất posterior là dưới 0.5. Không thể dễ hơn nữa. Nhưng bởi vì grid approximation không ứng dụng được trong thực tế, nó phức tạp hơn nhiều. Khi mà có nhiều hơn một parameter trong phân phối posterior (đợi đến chương sau để gặp được mô hình phức tạp này), ngay cả phép cộng đơn giản cũng không còn đơn giản nữa.

Vậy hãy xem cách nào để thực hiện phép tính này, bằng mẫu lấy từ posterior. Hướng tiếp cận này tổng quát hoá các mô hình phức tạp với nhiều parameter, và do vậy bạn có thể dùng nó ở mọi nơi. Những gì bạn cần là cộng hết những mẫu nào dưới 0.5, và chia nó cho tổng số lần lấy mẫu. Hay nói khác là, tìm tần suất của giá trị parameter nhỏ hơn 0.5:

<b>Code 3.7</b>
```python
jnp.sum(samples < 0.5) / 1e4
```
<samp>0.1711</samp>

Kết quả này gần giống với kết quả từ grid approximation cung cấp cho, mặc dù có thể không giống chính xác với kết quả của bạn, bởi vì mẫu mà bạn rút từ posterior chắc chắn sẽ khác với bản thân posterior. Vùng này được thể hiện ở biểu đồ trên bên trái ở [**HÌNH  3.2**](#f2). Sử dụng hướng tiếp cận này, bạn có thể hỏi xác suất posterior nằm giữa 0.5 và 0.75 là bào nhiêu:

<b>Code 3.8</b>
```python
jnp.sum((samples > 0.5) & (samples < 0.75)) / 1e4
```
<samp>0.6025</samp>

<a name="f2"></a>![](/assets/images/fig 3-2.svg)
<details class="fig"><summary>Hình 3.2: Hai loại khoảng posterior. Hàng trên: khoảng ranh giới xác định. Bên trái: vùng màu xanh là xác suất posterior nằm dưới parameter có giá trị là 0.5. Bên phải: Xác suất posterior nằm giữa 0.5 và 0.75. Hàng dưới: khoảng mật độ xác định. Bên trái: xác suất posterior chiếm 80% dưới nằm dưới một giá trị khoảng 0.75. Bên phải: xác suất posterior chiếm 80% giữa nằm ở khoảng 10% và 90%.</summary>
{% highlight python %}fig, axes = plt.subplots(2,2, figsize=(10,10))
for ax in axes:
    for sax in ax:
        sax.plot(p_grid, posterior)
        sax.set(xlabel="tỉ lệ nước (p)",
                ylabel="mật độ",
                xticks=[0,0.25, 0.5, 0.75, 1])
cond0 = p_grid<0.5
axes[0,0].fill_between(p_grid[cond0], posterior[cond0])
cond1 = (p_grid>0.5) & (p_grid<0.75)
axes[0,1].fill_between(p_grid[cond1], posterior[cond1])
cond2 = p_grid&lt;jnp.quantile(samples,0.8)
axes[1,0].fill_between(p_grid[cond2], posterior[cond2])
cond3 = (p_grid>jnp.quantile(samples,0.1))&(p_grid&lt;jnp.quantile(samples,0.9))
axes[1,1].fill_between(p_grid[cond3], posterior[cond3])
axes[1,0].annotate("80% dưới", (0.1, 0.002))
axes[1,1].annotate("80% giữa", (0.1, 0.002))
plt.tight_layout(){% endhighlight %}</details>

Khoảng 61% của xác suất posterior nằm ở giữa 0.5 và 0.75. Vùng này được thể hiện ở biểu đồ trên bên phải ở [**HÌNH 3.2**](#f2).

### 3.2.2 Khoảng mật độ xác định

Các tạp chí khoa học thường báo cáo giá trị trong khoảng mật độ cụ thể, gọi là **KHOẢNG TIN CẬY (CONFIDENCE INTERVAL)**. Còn khoảng mật độ trong xác suất posterior, được gọi là **KHOẢNG TÍN NHIỆM (CREDIBLE INTERVAL)**. Chứng ta sẽ gọi nó là **KHOẢNG THÍCH HỢP(COMPATIBILITY INTERVAL - CI)**, để tránh sự hiểu lầm của "confidence (tin cậy)" hay "credibility (tín nhiệm)".<sup><a name="r54" href="#54">54</a></sup> Khoảng mật độ là một khoảng các giá trị của parameter mà phù hợp với mô hình và data. Mô hình và data, cũng như khoảng mật độ, chúng không tạo ra sự tin vậy.

Khoảng posterior gồm 2 giá trị của parameter mà trong đó chứa mật độ xác suất được chọn. Với dạng câu hỏi này, ta sẽ dễ dàng hơn khi trả lời bằng mẫu từ posterior hơn là dùng toàn bộ posterior. Giả sử bạn muốn biết khoảng giá trị chứa 80% dưới xác suất posterior. Bạn biết khoảng này bắt đầu từ $p=0$. Để hỏi đến đâu nó dừng lại, hãy nghĩ mẫu như data và hỏi 80th percentile của nó nằm ở đâu:

<b>Code 3.9</b>
```python
jnp.quantile(samples, 0.8)
```
<samp>0.7637638</samp>

Vùng này được thể hiện ở biểu đồ dưới bên trái của [**HÌNH 3.2**](#f2). Tương tự, khoảng 80% giữa nằm ở khoảng 10th và 90th percentile. Biên giới này cũng được tìm ra bằng cách làm như trên, được thể hiện ở biểu đồ dưới phải của [**HÌNH 3.2**](#f2).

<b>Code 3.10</b>
```python
jnp.quantile(samples, [0.1, 0.9])
```
<samp>[0.44644645, 0.81681681]</samp>

Khoảng tin cậy với mật độ bằng nhau ở 2 đuôi rất thường gặp trong khoa học. Ta sẽ gọi nó là **KHOẢNG PERCENTILE (PERCENTILE INTERVAL - PI)**. Những khoảng này cho dùng để biểu diễn hình dạng của phân phối rất tốt, miễn là phân phối đừng có bất đối xứng quá. Nhưng trong suy luận giá trị parameter nào là kiên định nhất với data, nó không hoàn hảo. Hãy xem phân phối posterior và khoảng giá trị khác nhau ở [**HÌNH 3.3**](#f3). Posterior này kiên định với ba quan sát nước từ ba lần tung và có prior phẳng (đồng dạng). Nó rất lệch, có giá trị cực đại ở biên giới, $p=1$. Bạn có tính nó bằng grid approximation:

<b>Code 3.11</b>
```python
p_grid = jnp.linspace(start=0, stop=1, num=1000)
prob_p = jnp.repeat(1, 1000)
prob_data = jnp.exp(dist.Binomial(total_count=3, probs=p_grid).log_prob(3))
posterior = prob_data * prob_p
posterior = posterior / jnp.sum(posterior)
samples = p_grid[dist.Categorical(probs=posterior).sample(random.PRNGKey(0), (10000,))]
```

<a name="f3"></a>![](/assets/images/fig 3-3.svg)
<details class="fig"><summary>Hình 3.3: Sự khác nhau giữa percentile và mật độ xác suất lớn nhất. Mật độ posterior tương ứng với prior phẳng và ba mẫu nước quan sát được từ ba lần tung quả cầu. Trái: khoảng 50% percentile. Khoảng này gán khối lượng tương đương (25%) cho cả hai đuôi bên trái và bên phải. Kết quả là, nó bỏ qua giá trị hợp lý nhất, $p=1$. Phải: 50% khoảng mật độ posterior cao nhất, HDPI. Khoảng này tìm vùng hẹp nhất với 50% xác suất posterior. Vùng này luôn chứa giá trị hợp lý nhất.</summary>
{% highlight python %}fig, axes = plt.subplots(1,2, figsize=(10,4))
for ax in axes:
    ax.plot(p_grid, posterior)
    ax.set(xlabel="tỉ lệ nước (p)",
        ylabel="mật độ",
        xticks=[0,0.25, 0.5, 0.75, 1])
cond0 = (p_grid > jnp.quantile(samples,0.25))&(p_grid&lt;jnp.quantile(samples,0.75))
axes[0].fill_between(p_grid[cond0], posterior[cond0])
axes[0].set_title("Khoảng tin cậy 50% percentile")
cond1 = p_grid > jnp.quantile(samples,0.5)
axes[1].fill_between(p_grid[cond1], posterior[cond1])
axes[1].set_title("HPDI 50%")
plt.tight_layout(){% endhighlight %}</details>

Đoán code này cũng lấy mẫu trước từ posterior. Bây giờ, ở biểu đồ bên trái của [**HÌNH 3.3**](#f3), khoảng tin cậy 50% percentile được tô màu lên. Bạn có thể tính nó một cách dễ dàng thông qua hàm `jnp.quantile`:

<b>Code 3.12</b>
```python
jnp.percentile(samples, q=(25, 75))
```
<samp>[0.7077077, 0.93193191]</samp>

Khoảng này gán 25% mật độ trên và dưới khoảng, và 50% mật độ ở giữa. Nhưng trong ví dụ này, nó bỏ qua giá trị parameter thích hợp nhất gần $p=1$. Cho nên để mô tả hình dáng của posterior - là công việc của khoảng giá trị - bằng khoảng percentile có thể gây hiểu nhầm.

<div class="alert alert-info">
    <p><strong>Tại sao 95%?</strong> Khoảng mật độ được dùng nhiều nhất trong khoa học là 95%. Khoảng này cho 5% xác suất ở bên ngoài, tương ứng với 5% cơ hội parameter này không nằm trong trong khoảng. Truyền thống này phản ánh qua ngưỡng tin cậy 5% hay $p < 0.05$. Nhưng nó chỉ mang ý nghĩa thuận tiện. Ronald Fisher thường bị đổ tội cho lựa chọn này, qua bài viết năm 1925 nổi tiếng của ông:
    <blockquote>
        "[Giá trị của độ lệch chuẩn] mà tại đó $P= .05$, hay 1 trong 20, là 1.96 hay gần bằng 2; nó khá tiện khi chọn điểm này làm giới hạn để đánh giá một giá trị biến thiên khác có tin cậy hay không."<sup><a name="r55" href="#55">55</a></sup>
    </blockquote>
    Nhiều người không nghĩ rằng sự tiện lợi là một tiêu chí nghiêm túc. Sau này trong sự nghiệp của ông, Fisher chủ động khuyên ngăn dùng ngưỡng này để kiểm định mức tin cậy.<sup><a name="r56" href="#56">56</a></sup></p>
    <p>Vậy bạn cần làm gì? Không có chuẩn mực nào cả, nhưng có suy nghĩ là tốt. Nếu bạn cố gắng chứng minh khoảng tin cậy không chứa một giá trị nào đó, bạn hãy dùng khoảng tin cậy rộng nhất có thể mà loại giá trị đó ra. Thông thường, khoảng tin cậy được dùng để truyền đạt thông tin hình dáng của phân phối. Bạn tốt hơn hết là có nhiều khoảng tin cậy khác nhau. Trình bày khoảng 67%, 89%, 97%, kèm theo median được không? Tại sao lại là những giá trị này? Không lý do. Có lẽ vì nó là số nguyên tố và dễ nhớ hơn. Những gì quan trọng là nó đủ rộng để minh hoạ cho hình dáng của posterior. Và nên tránh giá trị 95%, bởi khoảng 95% tiện lợi đó kích gợi người đọc thực hiện những phép định một cách không tự chủ.</p>
</div>

Ngược lại, biểu đồ bên phải trong [**HÌNH 3.3**](#f3) là 50% **KHOẢNG MẬT ĐỘ LỚN NHẤT (HIGHEST POSTERIOR DENSITY INTERVAL - HDPI)**.<sup><a name="r57" href="#57">57</a></sup> HPDI là khoảng hẹp nhất chứa mật độ xác suất định trước. Nếu bạn nghĩ lại, nó có vô số các khoảng có cùng mật độ xác định. Nhưng nếu bạn muốn khoảng giá trị tốt nhất đại diện cho data, bạn cần khoảng mật độ đặc nhất. Nó là HPDI. Tính HPDI có thể dùng hàm sau:

<b>Code 3.13</b>
```python
numpyro.diagnostics.hpdi(samples, prob=0.5)
```
<samp> [0.8418418, 0.998999 ]</samp>

HPDI cho ta khoảng tin cậy chứa xác suất lớn nhất, cũng như hẹp hơn rất nhiều. Chiều rộng 0.16 nhỏ hơn 0.23 lúc nãy.

HPDI có lợi thế hơn khoảng percentile. Đa số trường hợp, hai khoảng này là gần bằng nhau.<sup><a name="r58" href="#58">58</a></sup> Nó khác nhau trong trường hợp này cho do phân phối posterior quá lệch. Nếu ta thay bằng mẫu từ posterior của sáu nước trong chín lần tung, hai khoảng này sẽ trùng với nhau. Bạn hãy thử với những mật độ xác suất khác nhau, như `prob=0.8` và `prob=0.9`. Khi posterior có hình quả chuông, việc dùng khoảng giá trị nào không còn ý nghĩa nữa. Nên nhớ rằng chúng ta không phải phóng tên lửa hay bắn hạt nhân, cho nên sự khác nhau ở số thập phân thứ 5 sẽ không cải thiện khoa học của bạn.

HPDI cũng có những bất lợi thế. HPDI thì tính toán nặng nhọc hơn so với khoảng percentile và bị ảnh hưởng bởi *sự biến thiên trong mô phỏng (simulation variance)*, nói cách khác tức là nó nhạy cảm với số lượng mẫu từ posterior. Nó cũng gây khó hiểu hơn và nhiều khán giá giới khoa học không thích công năng này, trong khi họ sẽ lập tức hiểu một khoảng tin cậy percentile, giống như trong khoảng non-Bayes thông thường được diễn giải (sai) như là một khoảng percentile.

Nhìn chung, nếu việc lựa chọn khoảng tin cậy gây ra sự khác biệt lớn, thì bạn không nên dùng khoảng tin cậy để mô tả posterior. Nhớ rằng, toàn bộ posterior là kết quả của "ước lượng Bayes". Nó mô tả tính phù hợp tương đối của mỗi giá trị có thể của parameter. Một khoảng giá trị trong phân phối chỉ có giá trị khi tóm tắt posterior. Nếu khoảng tin cậy bạn chọn ảnh hưởng nhiều đến suy luận, thì bạn tốt hơn hết là vẽ lại toàn bộ phân phối posterior.

<div class="alert alert-info">
    <p><strong>Khoảng tin cậy là gì?</strong> Người ta thường nói khoảng tin cậy 95% là xác suất 0.95 để giá trị thật nằm trong khoảng tin cậy. Trong suy luận thống kê non-Bayes, mệnh đề này là sai, vì suy luận non-Bayes không cho phép dùng xác suất để mô tả tính bất định của parameter. Ta phải nói, nếu ta lặp lại thí nghiệm này và phân tích với một số lượng lớn, thì 95% trong số lần đó sẽ ra khoảng tin cậy chứa giá trị thật. Nếu như bạn không rõ sự khác biệt này, thì bạn cũng giống như mọi người, bởi vì khái niệm này khá là trừu tượng, cho nên nhiều người diễn giải chúng giống như cách của Bayes.</p> 
    <p>Nhưng nếu bạn là Bayes hay không, thì khoảng 95% này không phải lúc nào cũng chứa giá trị thật 95% các trường hợp. Khoa học đã chứng minh khoảng tin cậy đã làm cho con người bị tự tin thái quá mạn tính.<sup><a name="r59" href="#59">59</a></sup> Từ "đúng" nên cảnh báo rằng có gì đó sai với mệnh đề như "chứa giá trị đúng". Con số 96% nằm trong <i>thế giới nhỏ</i>, nó chỉ đúng trong thế giới logic của mô hình. Nên nó sẽ không bao giờ được áp dụng chính xác tuyệt đối trong thế giới thực. Nó là những gì con golem tin tưởng, nhưng bạn được quyền tự do tin vào thứ khác. Nhưng cuối cùng, độ rộng của khoảng tin cậy, và giá trị nó chứa, có thể cung cấp được những thông tin tốt.</p>
</div>

### 3.2.3 Ước lượng điểm

Công việc mô tả thứ ba và cuối cùng của posterior là cho ra ước lượng điểm của một thứ gì đó. Với toàn bộ phân phối posterior, bạn nên báo cáo giá trị nào? Có thể đây là một câu hỏi ngây thơ, nhưng nó thực sự khó trả lời. Ước lượng parameter kiểu Bayes cho kết quả là toàn bộ posterior, chứ không phải một con số, mà là một hàm gán xác suất cho mỗi giá trị parameter. Cho nên quan trọng ở đây là bạn không cần phải đưa ra ước lượng điểm. Nó không cần thiết và có thể gây người đọc hiểu sai. Nó vứt thông tin đi.

Nhưng nếu bạn phải tạo ra một điểm ước lượng để mô tả posterior, bạn sẽ phải hỏi và trả lời nhiều câu hỏi hơn. Giả sử với ví dụ tung quả cầu, trong đó ta quan sát được 3 nước trong 3 lần tung, giống như [**HÌNH 3.3**](#f3). Bạn hãy xem xét ba cách ước lượng điểm sau. Đầu tiên, các nhà khoa học thường mô tả parameter có xác suất posterior cao nhất, ước lượng điểm *maximum a posteriori (MAP)*. Bạn có thể tính MAP dễ dàng trong ví dụ này:

<b>Code 3.14</b>
```python
p_grid[jnp.argmax(posterior)]
```
<samp>1.</samp>

Hoặc nếu bạn có mẫu từ posterior, bạn vẫn có thể ước lượng được điểm MAP:

<b>Code 3.15</b>
```python
samples[jnp.argmax(gaussian_kde(samples, bw_method=0.01)(samples))]
```
<samp>0.988989</samp>

Nhưng tại sao lại điểm này (còn gọi là mode), tại sao không phải trung bình (mean) hay trung vị (median) của posterior?

<b>Code 3.16</b>
```python
print(jnp.mean(samples))
print(jnp.median(samples))
```
<samp>0.8011085</samp>
<samp>0.8428428</samp>

Chúng cũng là những ước lượng điểm, và chúng cũng tóm tắt posterior. Nhưng cả ba điểm - mode (MAP), mean, median - đều khác nhau trong trường hợp này. Bạn chọn cái nào? [**HÌNH 3.4**](#f4) thể hiện phân phối posterior và vị trí của những ước lượng điểm này.

![](/assets/images/fig 3-4.svg)
<details class="fig"><summary>Hình 3.4: Ước lượng điểm và hàm mất mát. Bên trái: Phân phối posterior (xanh) sau khi quan sát được 3 nước trong 3 lần tung. Đường thẳng dọc là những vị trí của mode, median, mean (điểm mode, trung vị, trung bình). Mỗi ước lượng điểm suy ra hàm mất khác nhau. Bên phải: mất mát dự kiến dưới luật mà mất mát tỉ lệ với khoảng cách tuyệt đối từ quyết định (trục hoành) đến giá trị thực. Điểm tròn đánh dấu giá trị $p$ mà tối thiếu hoá mất mát dự kiến, là median của posterior.</summary>
{% highlight python %}fig, axes = plt.subplots(1,2, figsize=(9,4))
axes[0].plot(p_grid, posterior)
axes[0].set(xlabel="tỉ lệ nước (p)",
    ylabel="mật độ",
    xticks=[0,0.25, 0.5, 0.75, 1])
axes[0].vlines(samples.mean(),0, posterior.max(), linewidth=1)
axes[0].annotate("mean", (samples.mean()-0.05, 0.0005),rotation=90)
axes[0].vlines(jnp.quantile(samples, 0.5),0, posterior.max(), linewidth=1)
axes[0].annotate("median", (jnp.quantile(samples, 0.5), 0.001),rotation=90)
axes[0].vlines(p_grid[posterior.argmax()],0, posterior.max(), linewidth=1)
axes[0].annotate("mode", (p_grid[posterior.argmax()]-0.05, 0.0015),rotation=90)
d = jnp.linspace(0,1,1000)
fn = lambda p:jnp.sum(posterior*jnp.abs((d-p)))
loss = vmap(fn)(p_grid)
axes[1].plot(p_grid, loss)
axes[1].scatter(p_grid[jnp.argmin(loss)], loss.min(), s=100, c="r")
axes[1].set(xlabel="lựa chọn", ylabel="mất mất dự kiến tương ứng")
plt.tight_layout(){% endhighlight %}</details>

Một nguyên tắc khác ngoài cách dựa vào toàn bộ posterior là chọn một **HÀM MẤT MÁT (LOSS FUNCTION)**. Hàm mất mát là một luật lệ cho ta biết hao phí liên quan đến việc sử dụng một ước lượng điểm bất kỳ. Trong khi từ lâu các nhà thống kê đã hứng thú với hàm mất mát, và cách thống kê Bayes hỗ trợ chúng, các nhà khoa học ít có ai dùng nó một cách rõ ràng. Chìa khoá chính ở đây là *hàm mất mát khác nhau cho ước lượng điểm khác nhau.*

Ví dụ có một trò chơi, hãy nói cho tôi biết giá trị $p$ (tỉ lệ bề mặt nước ở quả cầu) nào, mà bạn nghĩ là đúng. Tôi sẽ trả bạn \\$100, nếu bạn trả lời chính xác điểm đó. Nhưng tôi sẽ trừ số tiền bạn được hưởng, tỉ lệ với khoảng cách từ lựa chọn của bạn đến giá trị đúng. Hay chính xác hơn, mất mát của bạn là tỉ lệ với giá trị tuyệt đối của $ d - p $, trong đó $d$ là lựa chọn của bạn và $p$ là câu trả lời đúng. Bạn có thể thay đổi trị giá chính xác của số tiền liên quan, mà không thay đổi nội dung quan trọng của vấn đề này. Cái quan trọng ở đay là mất mát tỉ lệ thuận với khoảng cách giữa lựa chọn của bạn đến giá trị thực.

Giờ khi bạn đã có posterior trong tay, bạn làm gì để bạn tối ưu hoá số tiền thắng. Thật vậy, giá trị mà bạn sẽ tối ưu hoá tiền thắng (tối thiểu hoá mất mát) chính là median của posterior. Ta hãy tính sự thật này, mà không cần chứng minh toán học. Ai muốn đọc bài chứng minh hãy đọc thêm endnote.<sup><a name="r60" href="#60">60</a></sup>

Cách tính mất mát dự kiến cho mỗi lựa chọn, là dùng posterior để trung bình hoá tính bất định của chúng ta về giá trị thực. Dĩ nhiên bạn không biết giá trị thực, trong đa số trường hợp. Nhưng nếu chúng ta dùng tất cả thông tin của mô hình về parameter, tức là chúng ta sẽ dùng toàn bộ phân phối posterior. Vậy giả sử bạn lựa chọn $p = 0.5$, thì mất mát dự kiến là:

<b>Code 3.17</b>
```python
jnp.sum(posterior * jnp.abs(0.5 - p_grid))
```
<samp>0.3128752</samp>

Ký hiệu `posterior` và `p_grid` là giống như những gì chúng ta đã làm trong chương này, lần lượt là xác suất posterior và các giá trị parameter. Đoán code trên tính mất mát trung bình có trọng số, ở đó mỗi giá trị mất mát được nhân với xác suất posterior tương ứng. Có một mánh để lặp lại phép tính này cho toàn bộ lựa chọn có thể, bằng hàm `jax.vmap`.

<b>Code 3.18</b>
```python
loss = vmap(lambda d: jnp.sum(posterior * jnp.abs(d - p_grid)))(p_grid)
```

Bây giờ ký hiệu `loss` chứa danh sách các giá trị mất mát, mỗi một giá trị cho mỗi lựa chọn khác nhau, tương ứng với giá trị trong `p_grid`. Từ đây, bạn có thể dễ dàng tìm giá trị parameter mà có mất mát thấp nhất là:

<b>Code 3.19</b>
```python
p_grid[jnp.argmin(loss)]
```
<samp>0.8408408</samp>

Đây thực ra chính là median của posterior, giá trị parameter mà chia xác suất posterior thành 2 phần có mật độ bằng nhau ở trên nó và dưới nó. Bạn có thể dùng hàm `jnp.median(samples)` để so sánh. Nó có thể không giống chính xác, do biến thiên của việc lấy mẫu, nhưng nó sẽ rất gần.

Vậy ta học được gì từ bài này? Để lựa chọn một ước lượng điểm, một giá trị đơn có thể tóm tắt được toàn bộ phân phối posterior, ta nên chọn một hàm mất mát. Hàm mất mát khác nhau thì cho ra ước lượng điểm khác nhau. Hai ví dụ phổ biến là giá trị tuyệt đối như trên, dẫn đến median thành ước lượng điểm, và hàm mất mát bậc hai $(d-p)^2$, dẫn đến mean của posterior của mẫu trở thành ước lượng điểm. Khi mà posterior đối xứng và nhìn giống normal, thì median và mean hội tụ lại thành chung một điểm, giúp ta thoải mái hơn vì không cần lựa chọn hàm mất mát. Trong bài toán gốc tung quả cẩu (6 nước trong 9 lần tung), mean và median chỉ khác nhau một ít.

Về mặt nguyên tắc, một ngữ cảnh có chi tiết áp dụng riêng có thể cần hàm mất mát riêng. Hãy xem một ví dụ thực hành giống như trong quyết định cần giải toả khu dân cư hay không, dựa vào tốc độ gió của bão. Đồ vật và sinh mạng sẽ thiệt hại rất nhanh khi tốc độ gió tăng. Lệnh giải toả vẫn có thể chi phí riêng, khi thực sự không cần thiết, nhưng mức độ ít hơn. Cho nên hàm mất mát này sẽ bất cân xứng, tăng mạnh khi tốc độ gió vượt dự đoán, mà tăng chậm khi tốc độ gió giảm xuống hơn dự đoán. Trong ví dụ này, ước lượng điểm cần thiết sẽ lớn hơn mean và median của posterior. Hơn nữa, vấn đề thực thiết là có nên giải toả hay không? Tạo ra ước lượng điểm cho tốc độ gió có thể không cần thiết.

Thông thường nhà nghiên cứu không quan tâm đến hàm mất mát. Các giá trị mean hoặc MAP trong báo cáo, không dùng cho mục đích củng cố lựa chọn nào cả, hoặc đơn thuần là mô tả hình dạng của posterior. Bạn có thể nói rằng lựa chọn ở đây là có từ chối hay chấp nhận giả thuyết. Nhưng thách thức sau đó là ta sẽ mất gì và được gì, sau khi chấp nhận hay từ chối giả thuyết.<sup><a name="r61" href="#61">61</a></sup> Thông thường ta nên trình bày toàn bộ posterior cùng với mô hình và data, để người khác có thể xây dựng ứng dụng trên tác phẩm của mình. Một lựa chọn chấp nhận hay từ chối giả thuyết quá non có thể ảnh hưởng tới sự sống khác.<sup><a name="r62" href="#62">62</a></sup>

Cho nên chúng ta luôn phải nhớ vấn đề này trong đầu, bởi nó giúp ta nhớ rằng những câu hỏi thường ngày trong suy luận thống kê chỉ có được trả lời dưới một bối cảnh cụ thể và mục đích cụ thể. Nhà thống kê có thể cung cấp dàn bài chung và tiêu chuẩn trả lời, nhưng một nhà khoa học nhiệt huyết luôn có thể cải thiện dựa trên những chỉ dẫn đó.

## <center>3.3 Lấy mẫu để mô phỏng dự đoán</center><a name="a3"></a>

Một công việc thường gặp với mẫu là mô phỏng quan sát mới của mô hình. Có ít nhất 4 lý do để làm việc này:

1. *Thiết kế mô hình.* Chúng ta có thể lấy mẫu không những từ posterior, mà còn từ prior. Khi nhìn mô hình hoạt động, trước và sau khi có data, là một cách để hiểu rõ hơn biểu hiện của prior. Chúng ta sẽ làm việc này rất nhiều, khi mà có nhiều parameter hơn và nên xác suất kết hợp của chúng không bao giờ rõ ràng.
2. *Kiểm tra mô hình.* Sau khi cập nhật data cho mô hình, ta cần mô phỏng mẫu quan sát tương lai từ suy luận của mô hình, để kiểm tra là mô hình fit đúng chưa và kiểm tra hành vi của mô hình.
3. *Kiểm tra phần mềm.* Để đảm bảo phần mềm hoạt động tốt, thì việc lấy mẫu mô phỏng so sánh với mô hình biết sẵn và sau đó tái hiện lại các giá trị của parameter mà data được mô phỏng ra.
4. *Thiết kế nghiên cứu.* Nếu bạn có thể mô phỏng mẫu từ giả thuyết, thì bạn có thể lượng giá hiệu quả của thiết kế nghiên cứu. Hay nói đơn giản là đây là *power analysis*, nhưng với tầm nhìn rộng hơn.
5. *Dự báo.* Ước lượng có thể dùng để dự đoán, cho trường hợp mới và quan sát trong tương lai. Những dự báo này có thể dùng trong thực dụng, cũng như đánh giá và nâng cấp mô hình.

Trong phần cuối của chương này, chúng ta sẽ xem các cách mô phỏng ra quan sát và làm sao để thực hiện vài thao tác kiểm tra đơn giản.

### 3.3.1 Data bù nhìn (dummy data)

Hãy tổng kết lại mô hình tung quả cầu. Quả cầu có tỉ lệ bề mặt nước $p$, là parameter đích mà ta đang suy luận. Số quan sát ra "nước" và "đất" tỉ lệ thuận lần lượt với $p$ và $1 - p$.

Chú ý rằng những giả định naỳ không chỉ cho chúng ta suy luận xác suất cho từng giá trị cụ thể của $p$ sau khi có quan sát. Nó còn cho phép ta mô phỏng quan sát mới từ mô hình. Nó cho phép vì hàm likelihood là hai chiều. Với một quan sát thực, hàm likelihood cho ta biết tình phù hợp của quan sát đó. Với một parameter, hàm likelihood cho ta phân phối của mọi quan sát mà ta có thể lấy từ phân phối, để tạo quan sát mới. Cho nên, mô hình Bayes luôn có tính *tái tạo (generative)*, dùng để mô phỏng quan sát. Rất nhiều mô hình non-Bayes có tính tái tạo, và cũng nhiều mô hình không có tính này.

Ta gọi data được mô phỏng này là **DATA BÙ NHÌN (DUMMY DATA)**, để chỉ rằng đây là một giá trị tạm cho data thực. Với ví dụ tung quả cầu, data bù nhìn từ likelihood binomial:

$$ \Pr(W|N,p) = \frac{N!}{W!(N-W)!} p^W (1-p)^{N-W} $$

Trong đó, $W$ là quan sát "nước" và $N$ là số lần tung. Giả sử $N=2$, 2 lần tung quả cầu. Thì có 3 khả năng xảy ra: 0 nước, 1 nước, 2 nước. Bạn có thể tính xác suất của từng khả năng, với giá trị $p$ bất kỳ. Ví dụ với $p=0.7$, tức là giá trị thực tỉ lệ nước của Trái Đất:

<b>Code 3.20</b>
```python
jnp.exp(dist.Binomial(total_count=2, probs=0.7).log_prob(jnp.arange(3)))
```
<samp>[0.08999996, 0.42000008, 0.48999974]</samp>

Có nghĩa là có 9% cơ hội để có quan sát $w=0$, 42% với $w=1$ và 49% với $w=2$. Khi thay đổi $p$ thì ta có kết quả khác.

Giờ ta sẽ mô phỏng các mẫu thử, bằng các xác suất này. Điều này thực thực hiện bằng hàm `sample` đã mô tả ở trên. 

<b>Code 3.21</b>
```python
dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(0))
```
<samp>1.</samp>


Sóp 1 có nghĩa là "1 nước trong 2 lần tung quả cầu". Ta có thể sample nhiều mẫu hơn. Một tập 10 mô phỏng có thể được thực hiện như sau:

<b>Code 3.22</b>
```python
dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(2), (10,))
```
<samp>[0., 2., 2., 2., 1., 2., 2., 1., 0., 0.]</samp>

Bây giờ tạo 100,000 data bù nhìn, và xem tần suất của mỗi giá trị (0,1,2) xuất hiện như thế nào trong tỉ lệ với likelihood của nó:

<b>Code 3.23</b>
```python
dummy_w = dist.Binomial(total_count=2, probs=0.7).sample(PRNGKey(0), (100000,))
jnp.unique(dummy_w, return_counts=True)[1] / 1e5
```
<samp>[0.0883 , 0.42101, 0.49069]</samp>

Những con số này gần giống với kết quả phân tích toán học trực tiếp từ likelihood. Bạn sẽ thấy giá trị hơi khác, do sự biến thiên của mô phỏng. Chạy đoạn code trên nhiều lần với `PRNGKey` khác nhau, để thấy được sự dao động tần suất khác nhau giữa mô phỏng này và mô phỏng kia.

Chỉ hai lần tung quả cầu thì không giống mẫu lắm. Giờ ta sẽ xem kết quả từ 9 lần tung.

<b>Code 3.24</b>
```python
dummy_w = dist.Binomial(total_count=9, probs=0.7).sample(random.PRNGKey(0), (100000,))
plt.hist(dummy_w, rwidth=0.4, bins=jnp.arange(11)-0.5);
```

<a name="f5"></a>![](/assets/images/fig 3-5.svg)
<details class="fig"><summary>Hình 3.5: Phân phối các các mẫu quan sát được mô phỏng từ 9 lần tung quả cầu. Những mẫu này có tỉ lệ nước giả định là 0.7.</summary>{% highlight python %}dummy_w = dist.Binomial(total_count=9, probs=0.7).sample(random.PRNGKey(0), (100000,))
plt.hist(dummy_w, rwidth=0.4, bins=jnp.arange(11)-0.5);
plt.xlabel("Số đếm nước bù nhìn")
plt.ylabel("tần số"){% endhighlight %}</details>

Kết quả sẽ ra như [**HÌNH 3.5**](#f5). Chú ý rằng là trong các mẫu quan sát thu được, đa số đều không cho tỉ lệ nước chính xác như $p = 0.7$. Đó là đặc tính của quan sát: mối quan hệ một-nhiều giữa data và mô hình xử lý tạo data. Bạn nên thí nghiệm với số lượng mẫu khác nhau, cũng như giá trị parameter khác nhau để xem sự thay đổi hình dáng và vị trí của phân phối của mẫu.

<div class="alert alert-info">
    <p><strong>Phân phối mẫu (Sampling distribution).</strong> Rất nhiều bạn đọc chắc là đã thấy mô phỏng mẫu quan sát. Phân phối mẫu là nền tảng của nhiều thống kê non-Bayes. Trong đó, suy luận parameter dựa trên phân phối mẫu này. Trong sách này, suy luận parameter không được làm trực tiếp trên phân phối mẫu. Posterior không phải được lấy mẫu, mà chúng ta tạo posterior theo quy trình logic. Và các mẫu quan sát được lấy từ posterior để hỗ trợ cho suy luận. Trong cả 2 trường hợp này, "lấy mẫu" không phải là hành động tồn tại vật lý. Trong hai trường hợp, nó chỉ là một thiết bị toán học và tạo ra các con số trong <i>thế giới nhỏ</i>.</p>
</div>

### 3.3.2 Kiểm tra mô hình

**KIỂM TRA MÔ HÌNH (MODEL CHECKING)** nghĩa là:
1. Đảm bảo mô hình hoạt động đúng sau khi fit.
2. Lượng giá sự thích hợp của mô hình với mục đích cụ thể.

Bởi vì mô hình Bayes luôn luôn có tính tái tạo, có thể mô phỏng quan sát cũng như ước lượng parameter từ các quan sát, khi bạn đặt điều kiện mô hình lên data, bạn có thể mô phỏng để kiểm tra kết quả đầu ra mong đợi của mô hình.

#### 3.3.2.1 Phần mềm có hoạt động đúng?

Trong trường hợp đơn giản nhất, ta chỉ kiểm tra phần mềm hoạt động đúng hay chưa bằng so sánh sự tương ứng giữa dự đoán và data để fit vào mô hình. Bạn có thể gọi đây là *dự đoán ngược*, tức là mô hình tái tạo ngược lại data dùng để fit mô hình có tốt không.  Nó không nhất thiết phải chính xác tuyệt đối. Nhưng khi không có sự tương ứng nào cả, thì có thể phần mềm bị lỗi.

Thật ra không có cách nào để đảm bảo phần mềm hoạt động đúng. Ngay cả phương pháp có dự đoán ngược giống với data quan sát, vẫn có thể không phát hiện được những lỗi nhỏ. Khi bạn làm việc với mô hình đa tầng, bạn sẽ thấy một hiện tượng dự đoán ngược không tương ứng với data quan sát. Mặc đó không có con đường hoàn hảo nào mà đảm bảo phần mềm hoạt động đúng, một phép kiểm tra đơn giản mà tôi khuyến khích ở đây thường sẽ bắt được những lỗi ngớ ngẩn, những lỗi mà con người hay mắc phải lần này qua lần khác.

Trong trường hợp phân tích ví dụ tung quả cầu, việc xây dựng phần mềm đơn giản đến nổi có thể kiểm tra bằng phân tích toán học. Cho nên ta sẽ đi tiếp theo để xem xét sự thích hợp của mô hình.

#### 3.3.2.2 Mô hình có thích hợp chưa?

Sau khi kiểm chứng phân phối posterior có đúng hay chưa, bằng một phần mềm hoạt động tốt, ta nên xem xét các khía cạnh của data mà mô hình không giải thích được. Mục đích không phải là kiểm tra giả định của mô hình là "đúng", bởi vì mọi mô hình đều là sai. Điều cần làm là tại sao mô hình lại thất bại trong việc mô tả data, để hướng tới hoàn thiện và nâng cấp mô hình.

Mọi mô hình đều có những thất bại ở vài khía cạnh nào đó, nên bạn phải biết tự mình - hoặc đồng nghiệp - phán đoán để nhìn nhận lỗi sai có quan trọng hay không. Một số người muốn tạo mô hình chỉ để mô tả lại mẫu thu thập được. Cho nên dự đoán (ngược) sai không phải là một điều xấu. Thông thường ta muốn dự đoán cho một quan sát tương lai và do đó hiểu rõ hơn để ta có thể tuỳ chỉnh thể giới thực. Ta sẽ xem xét những vấn đề này trong các chương sau.

Bây giờ, ta sẽ học cách kết hợp cách lấy mẫu quan sát giả tạo, với lấy mẫu các parameter từ posterior. Sẽ tốt hơn khi dùng toàn bộ phân phối posterior hơn là một ước lượng điểm nào đó. Tại sao? Bởi vì có một lượng lớn thông tin về tính bất định trong toàn bộ phân phối posterior. Ta sẽ mất thông tin ấy khi chỉ dùng một điểm nào đó và tính toán dựa trên nó. Chính sự mất mát thông tin này làm ta trở nên tự tin thái quá.

Ví dụ như trong mô hình tung quả cầu. Quan sát ở đây là số đếm của nước, trên số lần tung quả cầu. Dự đoán suy ra được mô hình là dựa trên tính bất định của 2 thành phần, và ta cần phải chú ý đến chúng.

Trước tiên là thành phần tính bất định từ quan sát. Với mỗi giá trị của $p$, có một kiểu tần suất các quan sát mà mô hình mong đợi, giống như mô hình khu vườn phân nhánh. Nó cũng là lấy mẫu từ likelihood mà bạn vừa mới thấy. Đây là yếu tố bất định trong dự báo quan sát mới, bởi vì khi bạn đã xác định $p$, bạn không biết lần tung sau như thế nào (nếu $p$ không phải 0 hay 1).

Thứ hai là thành phần tính bất định từ $p$ được biểu diễn bằng posterior. Chính vì sự bất định $p$ này, cho nên có sự bất định lên mọi thứ liên quan đến $p$. Và sự bất định này sẽ tương tác với sự bất định từ quan sát, khi ta cần tìm hiểu mô hình nói gì cho chúng ta về kết quả của nó.

Ta cần phải *lan truyền (propagation)* tính bất định từ parameter đến dự đoán, bằng cách lấy trung bình tất cả mật độ posterior cho $p$, khi tính ra dự đoán. Với mỗi giá trị của $p$, có một phân phối riêng. Và nếu bạn có thể tính được phân phối mẫu của kết cục tại mỗi điểm $p$, bạn có thể lấy trung bình tất cả phân phối dự đoán, bằng xác suất posterior, gọi là **phân phối dự đoán posterior (POSTERIOR PREDICTIVE DISTRIBUTION)**.

<a name="f6"></a>![](/assets/images/fig 3-6.svg)
<details class="fig"><summary>Hình 3.6: Mô phỏng dự đoán từ toàn bộ posterior. Trên: phân phối posterior quen thuộc của data tung quả cầu. Mười giá trị parameter ví dụ được đánh dấu bằng đường dọc. Giữa: Mỗi một giá trị parameter suy ra một phân phối mẫu độc nhất cho dự đoán. Dưới cùng: Tổng hợp tất cả các quan sát mô phỏng được thành một phần phối cho toàn bộ giá trị parameter (không chỉ mười cái trên), mỗi tập mô phỏng được đặt trọng số là xác suất posterior, tại ra phân phối dự đoán posterior. Phân phối này lan truyền tính bất định của parameter đến tính bất định về dự đoán.</summary>{% highlight python %}from matplotlib.patches import ConnectionPatch
fig = plt.figure(figsize=(15,8))
gs = fig.add_gridspec(3,9)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.edgecolor"] = "0.7"
plt.rcParams["axes.linewidth"]  = 2
plt.rcParams['patch.edgecolor'] = 'C0'
plt.rcParams['patch.facecolor'] = 'C0'
ax0 = fig.add_subplot(gs[0,:])
axes1 = [fig.add_subplot(gs[1,i]) for i in range(9)]
ax2 = fig.add_subplot(gs[2, 3:7])
tosses = jnp.arange(0,10)
ax0.plot(p_grid, posterior, label="Xác suất posterior")
ax0.set(
    xticks=[0.,0.5,1],
    yticks=[],
    xlabel="Tỉ lệ nước")
ax0.bar(p_grid[::10], posterior[::10], width=posterior[::10]/5)
ax0.legend()
for i in range(1,10):
    ax = axes1[i-1]
    ax.bar(
        tosses,
        jnp.exp(dist.Binomial(total_count=9, probs=i/10).log_prob(tosses)),
        width=0.5
    )
    ax.set(xticks=[],yticks=[], ylim=(0,0.4))
    x_pos = 7-i*0.8
    ax.text(x_pos, 0.35, f"0.{i}")
    plt.rcParams['patch.linewidth'] = max(posterior[::10][i]*100, 0.5)
    con = ConnectionPatch(xyA=(x_pos, 0.4), xyB=(i/10+0.001*i, 0),
                          coordsA="data", coordsB="data",
                          axesA=ax, axesB=ax0,
                          arrowstyle="-")
    ax.add_artist(con)
    con2 = ConnectionPatch(xyA=(x_pos, 0), xyB=(0.5, 1),
                          coordsA="data", coordsB="axes fraction",
                          axesA=ax, axesB=ax2,
                          arrowstyle="-")
    ax.add_artist(con2)
axes1[1].set_title("Phân phối mẫu")
ax2.hist(dist.Binomial(total_count=9, probs=samples).sample(random.PRNGKey(0)),
         rwidth=0.5, bins=jnp.arange(11)-0.5)
ax2.set(xticks=jnp.arange(0,11,3),yticks=[])
ax2.set_ylabel("Phân phối dự đoán\nposterior", rotation=0, labelpad=100, multialignment="right")
plt.subplots_adjust(hspace=0.5){% endhighlight %}</details>

[**HÌNH 3.6**](#f6) mô tả quy trình đã nêu. Trên cùng là posterior với 10 giá trị của parameter. Tại mỗi giá trị thì có một phân phối mẫu dự đoán riêng, được vẽ ở hàng giữa. Quan sát luôn có tính bất định cho một giá trị $p$ bất kỳ, nhưng sẽ thay đổi hình dạng tuỳ theo nó. Hàng cuối, là trung bình có trọng số của toàn bộ phân phối mẫu, sau khi dùng xác suất parameter từ posterior.

Kết quả cuối cùng là phân phối dự đoán posterior, nó lồng ghép tất cả tính bất định từ phân phối posterior của $p$. Cho nên, nó rất thật thà. Mặc dù mô hình đã làm rất tốt trong dự đoán data, những dự đoán vẫn còn khá loãng. Nếu bạn chỉ dùng một giá trị để tính dự đoán, ví dụ như dùng giá trị nằm ở đỉnh posterior, bạn sẽ tạo ra một phân phối của dự đoán mà bị tin cậy thái quá, nó hẹp so với phân phối dự đoán posterior ở [**HÌNH 3.6**](#f6) và sẽ thường lấy mẫu như hình $p=0.6$ ở hàng giữa. Hậu quả của sự tin cậy thái quá làm cho bạn có tin rằng mô hình này kiên định hơn với data hơn khả năng vốn dĩ của nó - tức là những dự đoán sẽ xoay quanh mẫu quan sát chặt hơn. Sai lầm trực giác này phát xuất từ việc bỏ đi tính bất định của parameter.

Vậy cụ thể tính như thế nào? Để mô phỏng quan sát dự đoán tự một giá trị $p$, như $p=0.6$, bạn có thể dùng `sample` từ phân phối binom.

<b>Code 3.25</b>
```python
w = dist.Binomial(total_count=9, probs=0.6).sample(random.PRNGKey(0), (int(1e4),))
```

Nó tạo ra 10,000 (1e4) dự đoán cho sự kiện 9 lần tung (`total_count=9`), giả định rằng $p=0.6$. Dự đoán này được lưu dưới dạng số đếm của nước, cho nên con số cực tiểu theo lý thuyết là zero, và con số cực đại là 9.

Bạn chỉ việc truyền tải tính bất định của parameter vào trong những dự đoán này bằng thay giá trị `0.6` thành mẫu từ posterior.

<b>Code 3.26</b>
```python
w = dist.Binomial(total_count=9, probs=samples).sample(random.PRNGKey(0))
```

`samples` ở trên là kết quả lấy mẫu $p$ từ posterior đã được tạo ở phần trước. Với mỗi giá trị trong `samples`, một quan sát từ binomial được tạo. Bởi vì `samples` tỉ lệ theo phân phối posterior, nên kết quả mô phỏng đã trung bình hoá posterior. Bạn có thể điều khiển những quan sát mô phỏng này giống như từng làm với phân phối posterior, như ước lượng khoảng và ước lượng điểm bằng quy trình tương tự. Nếu bạn vẽ biểu đồ những mẫu này, bạn sẽ có kết phân phối giống như biểu đồ bên phải của [**HÌNH 3.6**](#f6).

Kết quả từ mô phỏng dự đoán khá phù hợp với data quan sát được trong trường hợp này - số đếm 6 nằm ngay ở chính giữa của phân phối mô phỏng. Mẫu mô phỏng này khá loãng, do là nó từ bản thân quy trình lấy mẫu binomial, chứ không phải từ tính bất định của $p$. Nhưng còn khá non để kết luận mô hình này là hoàn hảo. Đến giờ, chúng ta chỉ nhìn data cũng giống như mô hình nhìn data vậy: Mỗi lần tung đều không liên quan với nhau. Giả định này cần xem xét lại. Nếu người tung quả cầu này là người cẩn thận, người đó có thể gây ra sự tương quan và có cùng kiểu kết quả trong chuỗi tung quả cầu. Hãy suy nghĩ nếu bề mặt quả cầu có một nửa là Thái Bình Dương. Như vậy, nước và đất không còn phân phối đồng dạng nữa, và nếu quả cầu không xoay đủ trong không trung, vị trí ban đầu khi tung có thể ảnh hưởng đến kết quả sau cùng. Vấn đề tưng tự xảy ra ở câu chuyện tung đồng xu, và thực vậy những người khéo léo có thể lợi dụng bản chất vật lý của đồng xu để ảnh hưởng đến kết quả tung.<sup><a name="r63" href="#63">63</a></sup>

Vậy với mục đích là tìm hiểu khía cạnh của dự đoán mà mô hình có thể sai, hãy nhìn data theo hai cách nhìn. Nhớ lại trình tự của chín lần tung là W L W W W L W L W. Trước tiên, hãy xem xét chuỗi liên tục dài nhất của nước hoặc đất. Nó cho ta ước lượng thô của tương quan giữa mỗi lần tung. Trong data quan sát, chuỗi liên tục dài nhất là 3 W. Thứ hai, xem xét số lần thay đổi giữa nước và đất. Đây cũng là một phương pháp đo lường tương quan giữa các mẫu. Trong data thì số lần thay đổi là 6. Không có gì đặc biệt trong 2 cách trình bày data này.

<a name="f7"></a>![](/assets/images/fig 3-7.svg)
<details class="fig"><summary>Hình 3.7: Cách nhìn khác về cùng một phân phối dự đoán posterior (xem <a href="f6"><strong>HÌNH 3.6</strong></a>. Thay vì nhìn data như mô hình đoán, tức là tổng số mẫu là nước, bây giờ ta nhìn data theo cách trình bày chuỗi liên tục dài nhất (trái) và số lần thay đổi giữa nước và đất (phải). Giá trị quan sát được tô màu đỏ. Trong khi dự đoán được mô phỏng thì kiên định với chuỗi liên tục (3 mẫu nước liên tục), chúng ít kiên định hơn với số lần thay đổi (6 lần thay đổi trong 9 lần tung)</summary>{% highlight python %}arr = dist.Binomial(1, samples).sample(random.PRNGKey(5), (9,))
def alter_view(arr):
    curr=0
    lngst=0
    switch=0
    for i in range(len(arr)-1):
        curr+=1
        if arr[i] != arr[i+1]:
            lngst=max(lngst, curr)
            curr = 0
            switch+=1
    else:
        curr+=1
    return max(lngst,curr), switch
runs = []
sws =[]
for col in range(arr.shape[1]):
    r,sw =alter_view(arr[:,col])
    runs.append(r)
    sws.append(sw)
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].hist(runs, rwidth=0.08, bins=jnp.arange(12)-0.5)
ax[0].bar(3, jnp.sum(jnp.array(runs)==3), color='r', width=0.2)
ax[0].set(xlabel="chuỗi liên tục dài nhất", ylabel="tấn số")
ax[1].hist(sws, rwidth=0.08, bins=jnp.arange(12)-0.5)
ax[1].bar(6, jnp.sum(jnp.array(sws)==6), color='r', width=0.2)
ax[1].set(xlabel="số lần thay đổi", ylabel="tấn số")
plt.tight_layout(){% endhighlight %}</details>

[**HÌNH 3.7**](#f7) là biểu đồ phân phối dự đoán posterior nhưng với 2 cách trình bày trên. Bên trái ta thấy chuỗi liên tục dài nhất của đât và nước của các mẫu đã lấy, và giá trị 3 được tô màu đỏ. Lần nữa, quan sát trong data trùng với quan sát mô phỏng, nhưng nó khá loãng. Bên phải, số lần thay đổi giữa nước và đất được vẽ lên, và giá trị 6 được tô màu đỏ. Bây giờ thì dự đoán được mô phỏng không đồng bộ với data, vì đa số mẫu dự đoán mô phỏng có số lần thay đổi ít hơn so với data mà chúng ta thu thập đươc. Điều này kiên định với sự thiếu hụt tính độc lập giữa ác lần tung quả cầu, có nghĩa là mỗi lần tung quả cầu thì tương quan âm với lần tung cuối.

Có phải mô hình của bạn không tốt? Tuỳ. Mọi mô hình có thể sai theo một khía cạnh nào đó. Nhưng điều này có dẫn ta đến thay đổi mô hình không thì còn tuỳ thuộc vào mục đích sử dụng. Trong trường hợp này, nếu việc tung quả cầu thường thay đổi từ W sang L hay L sang W, thì mỗi lần tung sẽ cho ít thông tin hơn tỉ lệ diện tích bề mặt. Trong thời gian dài, ngay cả mô hình sai mà ta dùng vẫn cho kết quả tỉ lệ bề mặt đúng. Nhưng nó sẽ làm chậm hơn so với phân phối posterior mà chúng ta dễ tin vào.

<div class="alert alert-info">
    <p><strong>Giá trị cực là gì?</strong> Một phương pháp thông thường để đo lường sự biến thiên của quan sát so với mô hình là đếm tần suất của vùng đuôi chứa data hiếm gặp hay giá trị cực. $p$-value thông thường là ví dụ của xác suất vùng đuôi này. Khi so sánh quan sát từ phân phối mẫu dự đoán được mô phỏng, như <a href="#f6"><strong>HÌNH 3.6</strong></a> và <a href="#f7"><strong>HÌNH 3.7</strong></a>, chúng ta thường hỏi vùng đuôi cách bao xa so với data quan sát được trước khi chúng ta kết luận mô hình là kém. Bởi vì có rất nhiều tình huống thống kê, không thể nào mà chỉ có một đáp án thoả mãn nhất.</p>
    <p>Có nhiều cách để định nghĩa "giá trị cực". $p$-value thông thường nhìn data theo kiểu mô hình mong muốn, nên nó là một dạng kiểm tra mô hình rất yếu. Ví dụ như hình bên phải cuối cùng trong <a href="#f6"><strong>HÌNH 3.6</strong></a> đánh giá mức độ fit mô hình theo cách tốt nhất cho mô hình. Cách định nghĩa khác của "giá trị cực" có thể là một thử thách nghiêm trọng cho mô hình. Những định nghĩa khác của giá trị cực như <a href="#f7"><strong>HÌNH 3.7</strong></a> có thể làm mô hình xấu hổ.</p>
    <p>Fit mô hình là một quy trình có tính đối tượng - mọi người và golem có thể cập nhật Bayes mà không phụ thuộc vào tuỳ thích cá nhân. Nhưng kiểm tra mô hình là chủ quan, và chính yếu tố này là điểm mạnh của mô hình Bayes, vì yếu tố chủ quan cần kinh nghiệm và kiến thức của người sử dụng. Người sử dụng có thể tưởng tượng nhiều cách kiểm tra năng suất của mô hình. Bởi vì golem không có trí tưởng tượng, chúng ta cần sự tự do để sử dụng sự tưởng tượng của mình. Theo cách này, mô hình Bayes có cả tính chủ quan và tính đối tượng.<sup><a name="r64" href="#64">64</a></sup></p>
</div>

## <center>3.4 Tổng kết</center><a name="a4"></a>

Chương này giới thiệu những quy trình cơ bản để xử lý phân phối posterior. Những công cụ cơ bản là mẫu các giá trị parameter được rút ra từ phân phối posterior. Làm việc với mẫu sẽ chuyển đổi bài tập đạo hàm tích phân thành bài tập tóm tắt data. Những mẫu này sẽ được dùng để tạo ước lượng khoảng, ước lượng điểm, phân phối dự đoán posterior, cũng như nhiều mô phỏng khác.

Phân phối dự đoán posterior kết hợp tính bất định về parameter, như được mô tả ở phân phối posterior, và tính bất định về kết cục, như mô tả qua hàm likelihood. Việc kiểm tra này là hữu ích cho việc xác định phần mềm hoạt động tốt. Chúng cũng giúp ích cho kiểm tra sự thích hợp mô hình.

Một khi mô hình phức tạp hơn, phân phối dự đoán posterior sẽ được dùng trong nhiều ứng dụng rộng rãi. Ngay cả việc hiểu mô hình cũng thường cần những quan sát từ mô phỏng. Chúng ta sẽ tiếp tục làm việc với các mẫu lấy từ posterior, để đơn giản hoá cũng như tuỳ biến hoá các tác vụ này khi có thể.

---

<details><summary>Endnotes</summary>
<ol class="endnotes">
<li><a name="50" href="#r50">50. </a>Gigerenzer and Hoffrage (1995). There is a large empirical literature, which you can find by searching forward on the Gigerenzer and Hoffrage paper.</li>
<li><a name="51" href="#r51">51. </a>Feynman (1967) provides a good defense of this device in scientific discovery.</li>
<li><a name="52" href="#r52">52. </a>For a binary outcome problem of this kind, the posterior density is given by <code>dbeta(p,w+1,n-w+1)</code>, where <code>p</code> is the proportion of interest, <code>w</code> is the observed count of water, and <code>n</code> is the number of tosses. If you’re curious about how to prove this fact, look up “beta-binomial conjugate prior.” I avoid discussing the analytical approach in this book, because very few problems are so simple that they have exact analytical solutions like this.</li>
<li><a name="53" href="#r53">53. </a>See Ioannidis (2005) for another narrative of the same idea. The problem is possibly worse than the simple calculation suggests. On the other hand, real scientific inference is more subtle than mere truth or falsehood of an hypothesis. I personally don’t like to frame scientific discovery in this way. But many, if not most, scientists tend to think in such binary terms, so this calculation should be disturbing.</li>
<li><a name="54" href="#r54">54. </a>I learned this term from Sander Greenland and his collaborators. See Amrhein et al. (2019) and Gelman and Greenland (2019).</li>
<li><a name="55" href="#r55">55. </a>Fisher (1925), in Chapter III within section 12 on the normal distribution. There are a couple of other places in the book in which the same resort to convenience or convention is used. Fisher seems to indicate that the 5% mark was already widely practiced by 1925 and already without clear justification.</li>
<li><a name="56" href="#r56">56. </a>Fisher (1956).</li>
<li><a name="57" href="#r57">57. </a>See Box and Tiao (1973), page 84 and then page 122 for a general discussion.</li>
<li><a name="58" href="#r58">58. </a>Gelman et al. (2013), page 33, comment on differences between percentile intervals and HPDIs.</li>
<li><a name="59" href="#r59">59. </a>See Henrion and Fischoff (1986) for examples from the estimation of physical constants, such as the speed of light.</li>
<li><a name="60" href="#r60">60. </a>Robert (2007) provides concise proofs of optimal estimators under several standard loss functions, like this one. It also covers the history of the topic, as well as many related issues in deriving good decisions from statistical procedures.</li>
<li><a name="61" href="#r61">61. </a>Rice (2010) presents an interesting construction of classical Fisherian testing through the adoption of loss functions.</li>
<li><a name="62" href="#r62">62. </a>See Hauer (2004) for three tales from transportation safety in which testing resulted in premature incorrect decisions and a demonstrable and continuing loss of human life.</li>
<li><a name="63" href="#r63">63. </a>It is poorly appreciated that coin tosses are very hard to bias, as long as you catch them in the air. Once they land and bounce and spin, however, it is very easy to bias them.</li>
<li><a name="64" href="#r64">64. </a>E. T. Jaynes (1922–1998) said all of this much more succinctly: Jaynes (1985), page 351, “It would be very nice to have a formal apparatus that gives us some ‘optimal’ way of recognizing unusual phenomena and inventing new classes of hypotheses that are most likely to contain the true one; but this remains an art for the creative human mind.” See also Box (1980) for a similar perspective.</li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
<p><strong>Easy.</strong> The Easy problems use the samples from the posterior distribution for the globe tossing example. This code will give you a specific set of samples, so that you can check your answers exactly.</p>
<b>code 3.27</b>
<pre>p_grid = jnp.linspace(start=0, stop=1, num=1000)
prior = jnp.repeat(1, 1000)
likelihood = jnp.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6))
posterior = likelihood * prior
posterior = posterior / jnp.sum(posterior)
samples = p_grid[dist.Categorical(posterior).sample(random.PRNGKey(100), (10000,))]</pre>
<p>Use the values in samples to answer the questions that follow.</p>
<p><strong>3E1.</strong> How much posterior probability lies below $p = 0.2$?</p>
<p><strong>3E2.</strong> How much posterior probability lies above $p = 0.8$?</p>
<p><strong>3E3.</strong> How much posterior probability lies between $p = 0.2$ and $p = 0.8$?</p>
<p><strong>3E4.</strong> 20% of the posterior probability lies below which value of $p$?</p>
<p><strong>3E5.</strong> 20% of the posterior probability lies above which value of $p$?</p>
<p><strong>3E6.</strong> Which values of $p$ contain the narrowest interval equal to 66% of the posterior probability?</p>
<p><strong>3E7.</strong> Which values of $p$ contain 66% of the posterior probability, assuming equal posterior probability both below and above the interval?</p>
<p><strong>3M1.</strong> Suppose the globe tossing data had turned out to be 8 water in 15 tosses. Construct the posterior distribution, using grid approximation. Use the same flat prior as before.</p>
<p><strong>3M2.</strong> Draw 10,000 samples from the grid approximation from above. Then use the samples to calculate the 90% HPDI for $p$.</p>
<p><strong>3M3.</strong> Construct a posterior predictive check for this model and data. This means simulate the distribution of samples, averaging over the posterior uncertainty in $p$. What is the probability of observing 8 water in 15 tosses?</p>
<p><strong>3M4.</strong> Using the posterior distribution constructed from the new (8/15) data, now calculate the probability of observing 6 water in 9 tosses.</p>
<p><strong>3M5.</strong> Start over at <strong>3M1</strong>, but now use a prior that is zero below $p = 0.5$ and a constant above $p = 0.5$. This corresponds to prior information that a majority of the Earth’s surface is water. Repeat each problem above and compare the inferences. What difference does the better prior make? If it helps, compare inferences (using both priors) to the true value $p = 0.7$.</p>
<p><strong>3M6.</strong> Suppose you want to estimate the Earth’s proportion of water very precisely. Specifically, you want the 99% percentile interval of the posterior distribution of $p$ to be only 0.05 wide. This means the distance between the upper and lower bound of the interval should be 0.05. How many times will you have to toss the globe to do this?</p>
<p><strong>Hard.</strong> The Hard problems here all use the data below. These data indicate the gender (male=1, female=0) of officially reported first and second born children in 100 two-child families.</p>
<b>code 3.28</b>
<pre>birth1 = [
    1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
]
birth2 = [
    0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
]</pre>
<p>So for example, the first family in the data reported a boy (1) and then a girl (0). The second family reported a girl (0) and then a boy (1). The third family reported two girls. You can load these two vectors into python's memory by typing:</p>
<b>code 3.29</b>
<pre>homeworkch3 = pd.read_csv("https://github.com/fehiepsi/rethinking-numpyro/blob/master/data/homeworkch3.csv?raw=true")</pre>
<p>Use these vectors as data. So for example to compute the total number of boys born across all of these births, you could use:</p>
<b>code 3.30</b>
<pre>sum(birth1) + sum(birth2)</pre>
<p><samp>111</samp></p>
<p><strong>3H1.</strong> Using grid approximation, compute the posterior distribution for the probability of a birth being a boy. Assume a uniform prior probability. Which parameter value maximizes the posterior probability?</p>
<p><strong>3H2.</strong> Using the <code>sample</code> function, draw 10,000 random parameter values from the posterior distribution you calculated above. Use these samples to estimate the 50%, 89%, and 97% highest posterior density intervals.</p>
<p><strong>3H3.</strong> Use <code>dist.Binomial</code> to simulate 10,000 replicates of 200 births. You should end up with 10,000 numbers, each one a count of boys out of 200 births. Compare the distribution of predicted numbers of boys to the actual count in the data (111 boys out of 200 births). There are many good ways to visualize the simulations, but the <code>az.plot_dist</code> command is probably the easiest way in this case. Does it look like the model fits the data well? That is, does the distribution of predictions include the actual observation as a central, likely outcome?</p>
<p><strong>3H4.</strong> Now compare 10,000 counts of boys from 100 simulated first borns only to the number of boys in the first births, <code>birth1</code>. How does the model look in this light?</p>
<p><strong>3H5.</strong> The model assumes that sex of first and second births are independent. To check this assumption, focus now on second births that followed female first borns. Compare 10,000 simulated counts of boys to only those second births that followed girls. To do this correctly, you need to count the number of first borns who were girls and simulate that many births, 10,000 times. Compare the counts of boys in your simulations to the actual observed count of boys following girls. How does the model look in this light? Any guesses what is going on in these data?</p>
</details>