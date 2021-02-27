---
title: "Chapter 5: The Many Variables & The Spurious Waffles"
description: "Chương 5: Các biến số & Xử sở Waffles giả tạo"
---

- [5.1 Tương quan giả tạo](#a1)
- [5.2 Tương quan bị ẩn](#a2)
- [5.3 Biến phân nhóm](#a3)
- [5.4 Tổng kết](#a4)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import daft
import matplotlib.pyplot as plt
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Một trong những nguồn waffle (bánh quế) đáng tin cậy nhất ở Bắc Mỹ, nếu không phải toàn bộ thế giới, là cửa hàng Waffle House. Cửa hàng luôn luôn mở cửa, ngay cả sau những cơn bão. Hầu như các cửa hàng của Waffle House đều trang bị cho dự phòng thiên tai, bao gồm cả máy phát điện riêng. Hậu quả là cơ quan dự báo thiên tai của Mỹ (FEMA) dùng luôn cửa hàng Waffle House làm chỉ số cho mức độ tàn phá của thiên tai.<sup><a name="r79" href="#79">79</a></sup> Nếu Waffle House phá sản, thì đó là một sự kiện nghiêm trọng.

Có một điều tức cười là Waffle House lại liên quan đến tỉ suất ly dị cao nhất của quốc gia [**HÌNH 5.1**](#f1). Bang nào có nhiều Waffle House theo mật độ, như Georgia và Alabama, có tỉ suất ly dị cao nhất nước Mỹ. Tỉ suất ly dị thấp nhất ở nơi không có Waffle House. Có lẽ nào bánh quế và khoai tây nghiền làm huỷ hoại hôn nhân?

Không hề. Đây là một ví dụ điển hình về mối tương quan gây hiểu nhầm. Không ai nghĩ ra được cơ chế hợp lý nào từ cửa hàng Waffle House gây ra ly dị. Thay vào đó, khi chúng ta thấy có một mối tương quan như vậy, ta lập tức phải tìm kiếm thêm các biến số khác mà gây ra sự hiểu lầm tai hại giữa bánh quế và ly dị. Trong trường hợp này, Waffle House mở ở bang Georgia vào năm 1955, và lan rộng khắp miền nam nước Mỹ và tồn tại chủ yếu ở đó. Cho nên Waffle House liên quan đến miền nam. Ly dị không phải đặc thù của miền nam nước Mỹ, nhưng ở đó có tỉ suất ly dị cao nhất cả nước. Chuyện Waffle House và tỉ suất ly dị ở miền nam có lẽ chỉ là một tại nạn lịch sử.

<a name="f1"></a>![](/assets/images/fig 5-1.svg)
<details class="fig"><summary>Hình 5.1: Số lượng của hàng Waffle House trên triệu người thì liên quan với tỉ suất ly dị (năm 2009) ở nước Mỹ. Mỗi điểm là một bang. Các bang miền Bắc là màu đỏ. Vùng tô màu là khoảng 89% của trung bình.</summary>
{% highlight python %}import seaborn as sns
x = d["WaffleHouses"]/d['Population']
y = d["Divorce"]
sns.regplot(x=x, y=y, ci=89, scatter=False)
sns.scatterplot(x=x,y=y, hue=d['South'], legend=False)
plt.gca().set(xlabel="Số cửa hàng Walffle House trên triệu người",
              ylabel="Tỉ suất ly dị",
              xlim=(-1,40))
for t in ['ME','AL','OK','AR','GA','SC','NJ']:
    cond = (d['Loc']==t)
    plt.annotate(t, (x[cond], y[cond])){% endhighlight %}</details>

Tai nạn như vậy rất thường gặp. Và nó cũng không đáng ngạc nhiên khi Waffle House tương quan với ly dị, bởi vì tổng quát thì tương quan là không gây ngạc nhiên. Trong tập data lớn, mọi cặp biến số vẫn có tương quan khác zero thấy rõ bằng thống kê.<sup><a name="r80" href="#80">80</a></sup> Nhưng bởi nhiều tương quan không phải là quan hệ nhân quả, chúng ta cần những công cụ để phân biệt giữa tương quan đơn thuần và bằng chứng nhân quả. Do đó mà có rất nhiều công sức đổ vào **HỒI QUY ĐA BIẾN (MULTIPLE REGRESSION)**, phương pháp sử dụng nhiều biến số để cùng lúc mô hình hoá biến kết cục. Những lý do cho hồi quy đa biến có thể kể đến:

1. "Kiểm soát" biến số gây nhiễu bằng thống kê. *Biến số gây nhiễu* là một thứ làm cho chúng ta hiểu nhầm về ảnh hưởng nhân quả - có rất nhiều định dạng chính xác hơn ở chương sau. Tương quan giả tạo (spurious correlation) giữa bánh quế và ly dị là một loại dạng nhiễu, khi mà việc nằm ở miền Nam làm cho biến số không có ảnh hưởng thực tế (mật độ Waffle House) lại trở nên quan trọng. Nhưng nhiễu rất đa dạng. Chúng có thể dễ dàng che đậy những hiệu ứng quan trọng cũng như tạo ra hiệu ứng sai.

2. Quan hệ nhân quả từ nhiều nguồn và phức tạp. Một hiện tượng có thể xuất phát từ nhiều nguyên nhân, và có thể lan truyền ra theo nhiều hướng phức tạp. Và bởi vì một nguyên nhân có thể che đậy nguyên nhân khác, chúng phải được đo lường cùng lúc.

3. Sự tương tác. Mức độ quan trọng của một biến có thể dựa vào biến số khác. Ví dụ, cây cối phát triển nhờ ánh sáng và nước. Nhưng không có một trong hai thứ, cây cối không nhận được gì hết. Sự **TƯƠNG TÁC (INTERACTION)** này thường xuyên xảy ra. Suy luận hiệu quả về một biến sẽ thường phụ thuộc vào sự xem xét của biến khác.

Trong chương này, chúng ta bắt đầu đối làm việc với hai lý do đầu tiên, sử dụng hồi quy đa biến để đối phó với những trường hợp nhiễu đơn giản và đo lường mối liên quan theo nhiều hướng. Bạn sẽ thấy cách để thêm một biến bất kỳ của *hiệu ứng chính* trong mô hình tuyến tính của trung bình Gaussian. Những hiệu ứng chính này là phép cộng của các biến số, loại đơn giản nhất của mô hình đa biến. Chúng ta sẽ tập trung vào hai thứ giá trị nhất mà những mô hình này có thể giúp chúng ta: (1) Phát hiện tương quan *giả tạo* như tương quan Waffle House và ly dị. (2) Phát hiện tương quan quan trọng được *che giấu* bằng tương quan bị ẩn với biến khác. Ở giữa bài, bạn sẽ gặp **BIẾN PHÂN NHÓM**, biến cần được xử lý đặc biệt so với biến liên tục.

Tuy nhiên, hồi quy đa biến có thể tệ hơn cả vô dụng, nếu chúng ta không biết sử dụng nó. Chỉ thêm biến vào mô hình có thể gây nhiều tổn thương. Trong chương này, chúng ta sẽ bắt đầu suy nghĩ về **SUY LUẬN NHÂN QUẢ** và giới thiệu sơ đồ nhân quả như là một cách để thiết kế và suy luận mô hình tuyến tính. Chương sau sẽ tiếp tục chủ đề này, mô tả những nguy hiểm nghiêm trọng và thường gặp trong việc thêm biến dự đoán, kết thúc bằng một khung quy trình thống nhất cho việc hiểu những ví dụ trong chương này và chương sau.

<div class="alert alert-info">
<p><strong>Suy luận nhân quả:</strong> Mặc dù nó có vai trò trung tâm, nhưng trong giới khoa học vẫn chưa thống nhất cách tiếp cận suy luận nhân quả. Thậm chí có người cho rằng nhân quả không hề tồn tại hoặc chỉ là một ảo tương về mặt tâm thần.<sup><a name="r81" href="#81">81</a></sup> Và trong hệ thống động phức tạp, mọi sự vật hiện tượng đều có thể là nguyên nhân của sự vật hiện tượng khác. "Nhân quả" mất đi giá trị trực quan của nó. Có một thứ, tuy nhiên, là một ý kiến được đồng thuận: Suy luận nhân quả luôn dựa vào những giả định chưa được kiểm chứng. Một cách nói khác là luôn luôn có thể tưởng tượng ra một tình huống để suy luận nhân quả của bạn là sai, cho dù có thiết kế hay phân tích rõ ràng cỡ nào. Song, vẫn có nhiều thứ có thể làm được, mặc dù có rào cản này.<sup><a name="r82" href="#82">82</a></sup></p></div>

## <center>5.1 Tương quan giả tạo</center><a name="a1"></a>

<a name="f2"></a>![](/assets/images/fig 5-2.svg)
<details class="fig"><summary>Hình 5.2: Tỉ suất ly dị liên quan với cả tỉ suất kết hôn (trái) và độ tuổi trung vị kết hôn (phải). Cả hai biến dự đoán được chuẩn hoá trong ví dụ này. Tỉ suất kết hôn trung bình giữa các bang là 20 trên 100 người trưởng thành, và độ tuổi kết hôn trung vị trung bình là 26 năm.</summary>
{% highlight python %}x1, x2 = d["Marriage"], d['MedianAgeMarriage']
y = d["Divorce"]
fig, axs = plt.subplots(1,2,figsize=(9,4))
sns.regplot(x=x1, y=y, ci=89, ax=axs[0])
sns.regplot(x=x2, y=y, ci=89, ax=axs[1])
axs[0].set(xlabel="Tỉ suất kết hôn", ylabel="Tỉ suất ly dị")
axs[1].set(xlabel="Độ tuổi kết hôn trung vị", ylabel="Tỉ suất ly dị"){% endhighlight %}</details>

Hãy bỏ bánh quế qua một bên, nhất là hiện tại. Một ví dụ dễ hiểu hơn là tương quan giữa tỉ suất ly dị và tỉ suất kết hôn.([**HÌNH 5.2**](#f2)) .Tỉ suất mà người trưởng thành kết hôn là một biến dự đoán tốt cho tỉ suất ly dị, như biểu đồ bên trái trong hình. Nhưng kết hôn có là *nguyên nhân* ly dị? Theo suy nghĩ bình thường thì nó rõ ràng đúng: bạn không thể ly dị nếu chưa kết hôn. Nhưng không có lý do gì để tỉ suất kết hôn cao phải gây ra nhiều cuộc ly dị. Có thể tưởng tượng ràng tỉ suất kết hôn cao chỉ điểm nền giá trị văn hoá tôn trọng giá trị của kết hôn, và do đó liên quan đến tỉ suất ly dị *thấp*.

Một biến dự đoán khác liên quan đến ly dị là độ tuổi trung vị khi kết hôn, thể hiện ở biểu đồ bên phải ở [**HÌNH 5.2**](#f2). Tuổi khi kết hôn cũng là một biến dự đoán tốt cho tỉ suất ly dị - khi độ tuổi lúc kết hôn tăng lên thì tỉ suất ly dị giảm xuống. Nhưng cũng không có lý do gì để nó là nhân quả, trừ phi bạn kết hôn ở độ tuổi khá trễ và bạn đời sống không đủ dài để đến ly dị.

Hãy tải data này và chuẩn hoá các biến quan tâm

<b>code 5.1</b>
```python
# load data and copy
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv?raw=true", sep=";")
# standardize variables
d["A"] = d.MedianAgeMarriage.pipe(lambda x: (x - x.mean()) / x.std())
d["D"] = d.Divorce.pipe(lambda x: (x - x.mean()) / x.std())
d["M"] = d.Marriage.pipe(lambda x: (x - x.mean()) / x.std())
```

Bạn có thể dựng lại biểu đồ bên phải bằng mô hình hồi quy tuyến tính:

$$\begin{aligned}
D_i &\sim \text{Normal}(\mu_i, \sigma)\\
µ_i & = \alpha + \beta_A A_i\\
\alpha &\sim \text{Normal}(0, 0.2)\\
\beta_A &\sim \text{Normal}(0, 0.5)\\
\sigma &\sim \text{Exponential}(1)\\
\end{aligned}$$

$D_i$ là tỉ suất ly dị ở bang $i$ được chuẩn hoá (trung tâm là zero, độ lệch chuẩn là một), và $A_i$ là độ tuổi kết hôn trung vị ở bang $i$ được chuẩn hoá. Cấu trúc mô hình tuyến tính giống như từ chương trước.

Prior thì sao? Bởi vì biến kết cục và biến dự đoán đều được chuẩn hoá, nên intercept $\alpha$ sẽ rất gần với zero. Còn slope $\beta_A=1$ suy ra gì? Khi $\beta_A = 1$, có nghĩa là khi thay đổi một đơn vị độ lệch chuẩn của độ tuổi kết hôn, sẽ tương ứng với thay đổi một đơn vị độ lệch chuẩn của ly dị. Để biết được độ mạnh của mối quan hệ này, bạn cần phải biết độ lệch chuẩn của độ tuổi kết hôn lớn như thế nào:

<b>code 5.2</b>
```python
d.MedianAgeMarriage.std()
```
<samp>1.24363</samp>

Khi $\beta_A = 1$, một sự thay đổi bằng 1.2 năm ở độ tuổi kết hơn thì liên quan với trọn một đơn vị độ lệch chuẩn của biến kết cục. Có vẻ đây là một tương quan mạnh bất thường. Prior ở trên nghĩ rằng có 5% giá trị phù hợp của slope là lớn hơn 1. Chúng ta sẽ mô phỏng từ những prior này ngay sau đây, để bạn có thể thấy được không gian kết cục từ chúng.

Để ước lượng posterior này, không có code hay kỹ thuật nào mới ở đây. Nhưng tôi sẽ thêm những comment để giải thích hàng loạt code sau.

<b>code 5.3</b>
```python
def model(A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bA * A)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
m5_1 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_1, optim.Adam(1), Trace_ELBO(), A=d.A.values, D=d.D.values)
p5_1, losses = svi.run(random.PRNGKey(0), 1000)
```

Để mô phỏng prior, chúng ta có thể dùng `Predictive` (không cần thêm đối số `posterior`)như chương trước. Tôi sẽ vẽ những đường thẳng trên khoảng 2 độ lệch chuẩn cho cả biến kết cục và biến dự đoán. Nó sẽ đảm bảo hầu hết các khoảng phù hợp của cả hai biến số.

<b>code 5.4</b>
```python
predictive = Predictive(m5_1.model, num_samples=1000, return_sites=["mu"])
prior_pred = predictive(random.PRNGKey(10), A=jnp.array([-2, 2]))
mu = prior_pred["mu"]
plt.subplot(xlim=(-2, 2), ylim=(-2, 2))
for i in range(20):
    plt.plot([-2, 2], mu[i], 'k', alpha=0.4)
```

<a name="f3"></a>![](/assets/images/fig 5-3.svg)
<details class="fig"><summary>Hình 5.3: Các đường hồi quy phù hợp được suy ra từ prior trong <code>m5.1</code>. Đây là những prior có thông tin yếu do đó chúng cho phép những mối quan hệ mạnh và phi lý nhưng nhìn chung giới hạn những đường thẳng đến khoảng khả thi của các biến số.</summary>{% highlight python %}predictive = Predictive(m5_1.model, num_samples=1000, return_sites=["mu"])
prior_pred = predictive(random.PRNGKey(10), A=jnp.array([-2, 2]))
mu = prior_pred["mu"]
plt.subplot(xlim=(-2, 2), ylim=(-2, 2))
for i in range(20):
    plt.plot([-2, 2], mu[i], color="C0",alpha=0.4)
plt.gca().set(xlabel="Độ tuổi kết hôn trung vị (chuẩn hoá)", ylabel="Tỉ suất ly dị (chuẩn hoá)"){% endhighlight %}</details>

[**HÌNH 5.3**](#f3) thể hiện kết quả trên. Bạn có thể thử những prior mơ hồ, phẳng hơn và xem những đường hồi quy prior trở nên lố bịch nhanh như thế nào.

Bây giờ đến dự đoán posterior. Quy trình này giống hệt với ví dụ chương trước: `Predictive`, tóm tắt bằng `jnp.mean` và `jnp.percentile`, và biểu đồ.

<b>code 5.5</b>
```python
# compute percentile interval of mean
A_seq = jnp.linspace(start=-3, stop=3.2, num=30)
post = m5_1.sample_posterior(random.PRNGKey(1), p5_1, (1000,))
post_pred = Predictive(m5_1.model, post)(random.PRNGKey(2), A=A_seq)
mu = post_pred["mu"]
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
# plot it all
az.plot_pair(d[["D", "A"]].to_dict(orient="list"))
plt.plot(A_seq, mu_mean, "k")
plt.fill_between(A_seq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
```

Nếu bạn kiểm tra `print_summary({x: post[x] for x in ['a', 'bA', 'sigma']}, 0.89,False)`, bạn sẽ thấy posterior của $\beta_A$ là một số âm đáng tin cậy, như trong [**HÌNH 5.2**](#f2).

Bạn có thể fit mô hình hồi quy tương tự cho quan hệ trong biểu đồ bên trái:

<b>code 5.6</b>
```python
def model(M, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bM * M
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
m5_2 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_2, optim.Adam(1), Trace_ELBO(), M=d.M.values, D=d.D.values)
p5_2, losses = svi.run(random.PRNGKey(0), 1000)
```

Như bạn có thể thấy trong hình, mối quan hệ này không mạnh như mối quan hệ trước.

Nhưng nếu chỉ đơn thuần so sánh trung bình của tham số giữa các hồi quy hai biến khác là không cách nào quyết định được biến dự đoán nào tốt hơn. Cả hai biến dự đoán đều cho những giá trị độc lập với nhau, hoặc chúng có thể là dư thừa, hoặc chúng sẽ tự loại trừ giá trị lẫn nhau.

Để hiểu đúng logic, chúng ta cần phải suy nghĩ nhân quả. Sau đó, chỉ sau khi chúng ta đã suy nghĩ, một mô hình hồi quy lớn hơn với cả hai độ tuổi khi kết hôn và tỉ suất ly dị sẽ cho có ích cho chúng ta.

### 5.1.1 Suy nghĩ trước khi hồi quy.

Có ba biến số quan sát được ở đây: tỉ suất ly dị ($D$), tỉ suất kết hôn ($M$), tuổi trung vị khi kết hôn ($A$) ở mỗi bang. Kết quả chúng ta thấy ở hai mô hình trước và được thể hiện trong [**HÌNH 5.2**](#f2) đều có vấn đề bởi vì chỉ một trong hai biến, trong trường hợp này là $A$, là có quan hệ nhân quả với biến kết cục, $D$, mặc dù cả hai biến số đều có tương quan mạnh với kết cục.

Để dễ hiểu hơn, chúng ta cần thêm một công cụ là sơ đồ nhân quả cụ thể có tên là **DAG** - viết tắt của **ĐỒ THỊ CÓ HƯỚNG KHÔNG TUẦN HOÀN (DIRECTED ACYCLIC GRAPH)**. Một đồ thị gồm các điểm (node) và cạnh(edge), thể hiện sự liên kết giữa các node. *Có hướng* nghĩa là các liên kết có mũi tên chỉ điểm cho ảnh hưởng nhân quả. Và *không tuần hoàn* nghĩa là nguyên nhân không dần dần chạy ngược về chính nó. DAG là một cách để mô tả định lượng các quan hệ nhân quả giữa các biến. Nó không chi tiết như mô tả đầy đủ toàn bộ mô hình, nhưng nó lại chứa thông tin mà mô hình thống kê không có được. Không giống như mô hình thống kê, DAG sẽ cho bạn biết hậu quả của việc can thiệp lên một biến số. Nhưng chỉ khi nào DAG đúng. Không có suy luận nào mà không có giả định.

Toàn bộ khung quy trình sử dụng DAG để thiết kế và đánh giá mô hình thống kê thì phức tạp. Cho nên thay vì ném bạn vào toàn bộ khung quy trình ngay bây giờ, tôi sẽ xây dựng nó bằng từng ví dụ một. Cuối chương sau, bạn sẽ có một tập các quy tắc đơn giản để hoàn thành việc đánh giá mô hình. Và sau đó những ứng dụng khác sẽ được giới thiệu trong chương sau đó.

Hãy bắt đầu bằng những thứ đơn giản. Đây là một DAG khả thi cho ví dụ tỉ suất ly dị của chúng ta:

![](/assets/images/dag 5-1.svg)

Nếu bạn muốn xem code để vẽ cái này, xem phần thông tin thêm ở cuối phần này. Nó có vẻ không nhiều, nhưng loại sơ đồ này làm rất nhiều việc. Nó đại điện cho mô hình nhân quả giúp phát hiện những vấn đề mới. Giống như tất cả mô hình khác, nó là một giả định về mặt phân tích. Ký hiệu $A$, $M$, và $D$ là biến quan sát của chúng ta. Mũi tên chỉ chiều hướng của sự ảnh hưởng. DAG này cho ta biết:
1. $A$ ảnh hưởng trực tiếp lên $D$.
2. $M$ ảnh hưởng trực tiếp lên $D$.
3. $A$ ảnh hưởng trực tiếp lên $M$.

Từ những mệnh đề này có thể suy ra nhiều kết quả mới. Trong trường hợp này, độ tuổi kết hôn có thể ảnh hưởng đến ly dị bằng hai cách. Một là từ hiệu ứng trực tiếp từ $A \to D$. Có lẽ hiệu ứng trực tiếp này đến từ sự việc người trẻ hay thay đổi hơn người lớn tuổi và hay trở nên không phù hợp với bạn đời của mình nữa. Hai là, hiệu ứng gián tiếp thông qua tỉ suất kết hôn, sau đó ảnh hưởng đến ly dị, $A \to M \to D$. Nếu người ta kết hôn sớm hơn, thì tỉ suất kết hôn tăng lên, bởi vì có nhiều người trẻ hơn. Giả sử có một chính sách ác độc ép mọi người kết hôn ở tuổi 65. Bởi vì chỉ có một bộ phận dân số sống đến 65 hơn 25, việc ép buộc hôn nhân trễ sẽ làm suy giảm tỉ suất kết hôn. Và nếu kết hôn có ảnh hưởng trực tiếp gì đến ly dị, có lẽ thông qua việc kết hôn được làm cho ít hay nhiều bình thường hơn, thì một phần ảnh hưởng trực tiếp đó có thể là hiệu ứng gián tiếp của độ tuổi kết hôn.

Để suy luận được cường độ của những mũi tên đó, chúng ta cần nhiều hơn một mô hình thống kê. Mô hình `m5.1`, hồi quy $D$ trên $A$, chỉ cho chúng ta biết *toàn bộ* hiệu ứng từ $A$ đến $D$ là tương quan phủ âm khá lớn với tỉ suất ly dị. *Toàn bộ* ở đây có nghĩa là mọi đường đi từ $A \to D$. Nó bao gồm hai đường trong DAG trên: $A \to D$, đường trực tiếp, $A \to M \to D$, đường gián tiếp. Nhìn chung, một biến số như biến $A$ có thể không có hiệu ứng trực tiếp gì cả đến kết cục như biến $D$. Nó có thể ảnh hưởng đến $D$ thông qua hoàn toàn bằng con đường gián tiếp. Kiểu quan hệ này còn gọi là **ĐIỀU CHỈNH (MEDIATION)**, và chúng ta sẽ có ví dụ cho nó.

Như bạn sẽ được thấy, tuy nhiên, con đường gián tiếp sẽ không có vai trò gì trong trường hợp này. Làm sao để thể hiện điều đó? Ta biết rằng mô hình `m5.2` có tương quan dương giữa tỉ suất kết hôn và tỉ suất ly dị. Nhưng nó không đủ để nói rằng con đường $M \to D$ là tương quan dương. Có thể toàn bộ tương quan giữa $M$ và $D$ hoàn toàn đến từ ảnh hường của $A$ vào cả hai $D$ và $M$. Giống như vậy:

![](/assets/images/dag 5-2.svg)

DAG này cũng cho kết quả kiên định với phân phối posterior của mô hình `m5_1` và `m5_2`. Tại sao? Bởi vì cả $M$ và $D$ đều "nghe" theo $A$. Chúng đều có thông tin từ $A$. Cho nên khi bạn kiểm tra tương quan giữa $D$ và $M$, bạn sẽ nhận được tín hiệu chung mà cả hai nghe từ $A$. Bạn sẽ thấy một phương pháp chính thống để kết luận điều này, ở chương sau.

Vậy DAG nào là đúng? Có tồn tại hiệu ứng trực tiếp từ tỉ suất kết hôn, hay là độ tuổi kết hôn chỉ tác động lên chúng, để tạo ra tương quan giả tạo giữa tỉ suất kết hôn và tỉ suất ly dị? Để tìm hiểu vấn đề này, chúng ta cần xem xét kỹ lưỡng những DAG ấy gợi ý điều gì. Và đó là phần tiếp theo.

<div class="alert alert-info">
<p><strong>Nhân quả là gì?</strong> Câu hỏi nhân quả là một trong những câu hỏi tốn nhiều giấy mực nhất trong triết học. Nhưng những cuộc tranh luận ấy không liên quan nhiều đến khoa học thống kê. Trong thống kê, khi biết được nguyên nhân, chúng ta có thể dự đoán đúng hậu quả của một hành động can thiệp. Tuy nhiên có nhiều trường hợp khá phức tạp. Ví dụ, bạn không thể thay đổi trực tiếp trọng lượng của một người. Thay đổi trọng lượng của một ai đó sẽ là can thiệp trên một biến khác, như thay đổi chế độ ăn, và biến đó sẽ cho thêm những hiệu ứng nhân quả khác đi kèm. Nhưng nhẹ cân vẫn là một chỉ điểm phù hợp của bệnh lý, mặc dù chúng ta không thể can thiệp trực tiếp lên đó được.</p></div>

<div class="alert alert-dark">
<p><strong>Vẽ DAG.</strong> Có nhiều cách để vẽ và phân tích DAG. Trong sách này, chúng ta sẽ sử dụng <code>daft</code>, <code>causalgraphicalmodels</code> và <code>daggity</code>. <code>daft</code> là một package python để vẽ sơ đồ nhân quả. <code>causalgraphicalmodels</code> dùng để phân tích sơ đồ nhân quả. <code>daggity</code> là một package trong R và có thể dùng được trên trang web [http://www.daggity.net/](http://www.daggity.net/). Để vẽ DAG đơn giản bạn thấy trong phần này:</p>
{% highlight python %}dag5_1 = CausalGraphicalModel(
    nodes=["A", "D", "M"], edges=[("A", "D"), ("A", "M"), ("M", "D")])
pgm = daft.PGM()
coordinates = {"A": (0, 0), "D": (1, 1), "M": (2, 0)}
for node in dag5_1.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_1.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis(){% endhighlight %}
<p>Thứ tự trong <code>edges</code> cũng là chiều hướng của sự ảnh hưởng. <code>coordinates</code> cho phép tuỳ chỉnh vị trí các node theo ý thích của bạn.</p></div>

### 5.1.2 Gợi ý kiểm tra được

Làm sao dùng data để so sánh nhiều mô hình phù hợp khác nhau? Việc đầu tiên là xem xét các **GỢI Ý KIỂM TRA ĐƯỢC (TESTABLE IMPLICATION)** của từng mô hình. Xem xét hai DAG chúng ta đang có:

![](/assets/images/dag 5-3.svg)

DAG bất kỳ có thể gợi ý biến số nào là độc lập với những biến khác dưới những điều kiện nhất định. Chúng là những gợi ý kiểm tra được của mô hình, hay những mối quan hệ **ĐỘC LẬP CÓ ĐIỀU KIỆN (CONDITIONAL INDEPENDENCIES)**. Quan hệ độc lập có điều kiện có hai hình thái. Đầu tiên, chúng là mệnh đề về các biến nào nên có (hoặc không có) mối quan hệ với một biến khác trong data. Thứ hai, chúng là mệnh đề về các biến nào sẽ bị cắt đứt mối quan hệ nếu chúng ta đặt điều kiện trên tập biến khác.

"Điều kiện" ở đây nghĩa là gì? Nói đơn giản, đặt điều kiện lên biến $Z$ nghĩa là học giá trị của nó và sau đó hỏi rằng nếu biết thêm $X$ thì có cho thêm thông tin gì vào $Y$ không. Nếu biết thêm $X$ không cho thêm thông tin vào $Y$, thì chúng ta có thể nói $Y$ độc lập với $X$ với điều kiện trên $Z$. Mệnh đề điều kiện này thường được viết dưới dạng: $Y \perp\\!\\!\perp X \| Z$. Đây là một ký hiệu khá lạ và cảm giác khó chịu đó của bạn đã được chứng minh. Chúng ta sẽ làm việc với khái niệm này rất nhiều, cho nên đừng lo lắng nếu bây giờ bạn chưa hiểu nó. Bạn sẽ sớm thấy nhiều ví dụ về nó.

Hãy xem xét những mối quan hệ độc lập có điều kiện trong ví dụ ly dị. Mối quan hệ độc lập có điều kiện ở hai DAG trên là gì? Làm sao chúng ta tìm ra được câu trả lời? Tìm ra mối quan hệ độc lập có điều kiện là không khó, những cũng không phải lúc nào cũng rõ ràng. Với một chút luyện tập, nó sẽ rất dễ. Nguyên tắc chung thì cần phải đợi đến chương tiếp theo. Còn bây giờ, hãy xem xét lần lượt từng DAG và kiểm tra các khả năng.

Với DAG bên trái, DAG có ba mũi tên, đầu tiên chú ý rằng mọi cặp biến đều có tương quan với nhau. Đó là bởi vì giữa chúng luôn có mũi tên nhân quả. Những mũi tên thì tạo qua tương quan. Vậy trước khi ta đặt điều kiện lên bất cứ thứ gì, mọi biến đều liên quan toàn bộ biến còn lại. Đây đã là một gợi ý kiểm tra được. Chúng ta có thể viết lại như sau:

$$ D \perp\!\!\!\not{}\!\!\!\perp A \quad D\perp\!\!\!\not{}\!\!\!\perp M \quad A \perp\!\!\!\not{}\!\!\!\perp M $$

Dấu $\perp\\!\\!\\!\not{}\\!\\!\\!\perp$ nghĩa là "không độc lập". Nếu chúng ta khám phá data và thấy có cặp biến nào không tương quan, thì DAG đó là không đúng. Trong data này, cả ba cặp thực ra đều tương quan với nhau. Hãy tự kiểm tra điều đó. Bạn có thể dùng `d[['A','D','M']].corr()` để đo lường những tương quan đơn giản. Tương quan đôi khi là cách đo lường mối liên quan rất tệ - nhiều dạng liên quan khác nhau với gợi ý khác nhau có thể tạo ra tương quan giống nhau. Những trong trường hợp này nó hoạt động tốt.

DAG đầu tiên còn gợi ý kiểm tra được nào không? Không, sẽ dễ hơn để thấy rằng, nếu chúng ta xem xét DAG thứ hai, trong đó $M$ không ảnh hưởng lên $D$. Trong DAG này, cả ba biến đều có liên quan với biến còn lại. $A$ liên quan đến $D$ và $M$ bởi vì nó ảnh hưởng cả hai. Và $D$ và $M$ liên quan với nhau, bởi vì $A$ ảnh hưởng cả hai. Chúng có cùng một nguyên nhân, và điều này dẫn đến chúng tương quan với nhau qua nguyên nhân đó. Nhưng giả sử chúng ta đặt điều kiện lên A. Tất cả những thông tin trong $M$ cho dự đoán $D$ là ở trong $A$. Cho nên nếu chúng ta đặt điều kiện lên $A$, $M$ không nói gì thêm về $D$. Cho nên ở DAG thứ hai, có thêm một gợi ý kiểm tra được là $D$ độc lập với $M$, đặt điều kiện trên $A$. Nói một cách khác, $D \perp\\!\\!\perp M \| A$. Chuyện tương tự sẽ không xảy ra với DAG thứ nhất. Đặt điều kiện trên $A$ sẽ không làm cho $D$ độc lập với $M$, bởi vì $M$ thực sự ảnh hưởng lên $D$ bởi chỉ mình nó trong mô hình này.

Trong chương sau, tôi sẽ giới thiệu nguyên tắc chung để cho ra những gợi ý này. Bây giờ, package `causalgraphicalmodels` đã có hàm giúp bạn tìm ra những gợi ý đó. Đây là code định nghĩa DAG thứ hai và mối quan hệ độc lập có điều kiện của nó.

<b>code 5.8</b>
```python
DMA_dag2 = CausalGraphicalModel(nodes=["A", "D", "M"], edges=[("A", "D"), ("A", "M")])
DMA_dag2.get_all_independence_relationships()
```
<samp>('M', 'D', {'A'})</samp>

DAG thứ nhất không có mối quan hệ độc lập có điều kiện. Bạn có thể định nghĩa nó và kiểm tra lại bằng:

<b>code 5.9</b>
```python
DMA_dag1 = CausalGraphicalModel(
    nodes=["A", "D", "M"], edges=[("A", "D"), ("A", "M"), ("M", "D")]
)
DMA_dag1.get_all_independence_relationships()
```

DAG này không những mối quan hệ độc lập có điện kiện, cho nên không có kết quả hiện ra.

Hãy thử tóm tắt lại. Gợi ý kiểm tra đợi từ DAG đầu tiên là tất cả các cặp biến số đều có quan hệ với nhau, cho dù chúng ta đặt điều kiện trên đâu. Gợi ý có điều kiện của DAG thứ hai là tất cả các cặp đều có liên quan, trước khi đặt điều kiện trên bất cứ gì, nhưng $D$ và $M$ sẽ độc lập sau khi đặt điều kiện trên $A$. Cho nên gợi ý duy nhất khác nhau giữa hai DAG là cái cuối cùng: $D \perp\\!\\!\perp M \| A$.

Để kiểm tra gợi ý này, chúng ta cần một mô hình thống kê có đặt điều kiện trên $A$, để chúng ta có thể thấy $D$ có độc lập với $M$ hay không.
Và thứ giúp đỡ chúng ta là hồi quy đa biến. Nó có thể trả lời cho câu hỏi *mô tả* hữu ích này:

>Có giá trị mới nào khi biết thêm một biến, một khi tôi đã biết tất cả những biến dự đoán khác?

Cho nên ví dụ rằng một khi bạn fit mô hình đa biến để dự đoán tỉ suất ly dị, sử dụng cả tỉ suất kết hôn và độ tuổi kết hôn, mô hình trả lời cho các câu hỏi:
1. Sau khi tôi đã biết tỉ suất kết hôn, có giá trị mới nào khi cũng biết thêm độ tuổi kết hôn?
2. Sau khi tôi đã biết độ tuổi kết hôn, có giá trị mới nào khi cũng biết thêm tỉ suất kết hôn?

Tham số ước lượng tương ứng với mỗi biến dự đoán là câu trả lời (thường bị che đậy) cho những câu hỏi này. Những câu hỏi trên là có tính mô tả, và những câu trả lời cũng là có tính mô tả. Nó là kết quả duy nhất từ những gợi ý kiểm tra được ở trên mà cho những mô tả có ý nghĩa nhân quả. Những ý nghĩa đó vẫn phải phụ thuộc vào lựa chọn DAG.

<div class="alert alert-info">
<p><strong>"Kiểm soát" bị mất kiểm soát.</strong> Thông thường câu hỏi ở trên là đại diện "kiểm soát bằng thống kê, như trong <i>kiểm soát</i> hiệu ứng của một biến khi ước lượng hiệu ứng của biến khác. Nhưng đây là một ngôn ngữ nửa vời, vì nó suy ra quá nhiều thứ. Kiểm soát thông kê khá là khác với kiểm soát bằng thực nghiệm, như chúng ta sẽ khám phá thêm ở chương sau. Vấn đề ở đây không phải là ngôn từ lịch sự. Thay vào đó, điểm chính là phải quan sát sự khác nhau giữa diễn giải thế giới thực và thế giới lớn. Bởi vì nhiều người dùng thống kê không phải là nhà thống kê, ngôn ngữ nửa vời như "kiểm soát" có thể khuyến khích một văn hoá diễn giải nửa vời. Và văn hoá này có xu hướng ước lượng thái quá sức mạnh của các phương pháp thống kê, cho nên ngăn cản chúng có thể rất khó khăn. Tự rèn luyện ngôn ngữ của bạn có lẽ là đủ. Rèn luyện ngôn ngữ của ai khác thì rất khó, để không bị giống như đang la mắng, như phần thông tin này.</p></div>

### 5.1.3 Ký hiệu hồi quy đa biến

Công thức hồi quy đa biến nhìn rất giống với mô hình đa thức ở cuối chương trước - chúng thêm vào nhiều tham số và biến số để định nghĩa $\mu_i$. Chiến lược khá là rõ ràng:
1. Liệt kê các biến dự đoán muốn dùng trong mô hình tuyến tính của trung bình.
2. Với mỗi biến dự đoán, tạo ra một tham số để giúp đo lường mối liên quan có điều kiện với biến kết cục.
3. Nhân tham số với biến số và thêm số hạng đó vào mô hình tuyến tính.

Ví dụ luôn luôn là cần thiết, cho nên đây là mô hình dùng để dự đoán tỉ suất ly dị, sử dụng cả độ tuổi kết hôn và tỉ suất kết hôn:

$$ \begin{aligned}
D_i &\sim \text{Normal}( \mu_i, \sigma) \quad && [\text{xác suất của data}]\\
\mu_i &= \alpha + \beta_M M_i + \beta_A A_i \quad && [\text{mô hình tuyến tính}]\\
\alpha &\sim \text{Normal}(0, 0.2) \quad && [\text{prior cho } \alpha]\\
\beta_M &\sim \text{Normal}(0, 0.5) \quad && [\text{prior cho } \beta_M]\\
\beta_A &\sim \text{Normal}(0, 0.5) \quad && [\text{prior cho } \beta_A]\\
\sigma &\sim \text{Exponential}(1) \quad && [\text{prior cho } \sigma]\\
\end{aligned} $$

Bạn có thể sử dụng bất kỳ ký hiệu nào bạn thích cho tham số và biến số, nhưng ở đây tôi chọn $M$ cho tỉ suất kết hôn và $A$ cho độ tuổi kết hôn, tái sử dụng những ký hiệu đó cho tham số tương ứng. Nhưng bạn được tự do sử dụng những ký hiệu giúp giảm nhẹ trọng tải bộ nhớ của chính bạn.

Vậy có ý nghĩa gì khi giả định $\mu_i = \alpha + \beta_M M_i + \beta_A A_i$? Theo cơ chế, nó có nghĩa là kết cục mong đợi cho bất kỳ bang nào có tỉ suất kết hôn $M_i$ và độ tuổi kết hôn trung vị $A_i$ là tổng của ba số hạng độc lập. Nếu bạn giống như bao người khác, điều này còn khá bí ẩn. Ý nghĩa theo cơ chế của phương trình không thể chuyển qua một ý nghĩa nhân quả độc nhất. Hãy chăm sóc cho phần cơ chế này, trước khi quay lại để diễn giải.

<div class="alert alert-dark">
<p><strong>Ký hiệu rút gọn và ma trận thiết kế.</strong> Thông thường, mô hình tuyến tính được viết dưới dạng rút gọn:</p>
$$\mu_i = \alpha + \displaystyle\sum_{j=1}^n \beta_jx_{ji}$$
<p>$j$ là chỉ điểm trên các biến dự đoán và $n$ là số các biến dự đoán. Nó có thể được đọc lại là <i>trung bình được mô hình hoá theo tổng của một intercept và tổng của tích tham số với biến dự đoán.</i> Gọn hơn nữa, bằng cách dùng ký hiệu ma trận:</p>
$$ m = Xb $$
<p>Trong đó $m$ là vector các trung bình dự đoán được, mỗi một cho từng dòng trong data, $b$ là một (cột) vector các tham số, mỗi một cho từng biến dự đoán, và $X$ là một ma trận. Ma trận này gọi là <i>ma trận thiết kế (design matrix)</i>. Nó có số dòng như data, và số cột như số lượng các biến dự đoán cộng một. Vậy $X$ cơ bản là một DataFrame, nhưng dư một cột đầu tiên. Cột dư đó chứa đầy các số 1. Những số 1 này được nhân với tham số đầu tiên, tức là intercept, và nên nó trả về intercept không chỉnh sửa. Khi $X$ được nhân ma trận vởi $b$, bạn nhận về các trung bình được dự đoán. Trong python, phép toán này là <code>X@b</code>.</p>
<p>Chúng ta sẽ không sử dụng cách tiếp cận ma trận thiết kế. Nhưng cũng tốt nếu bạn nhận ra nó, và đôi khi nó giúp bạn tiết kiệm rất nhiều sức lực. Ví dụ, trong hồi quy tuyến tính, có công thức ma trận rất hay cho ước lượng likelihood cực đại (hoặc <i>bình phương nhỏ nhất (least square)</i>). Đa số các phần mềm thống kê sử dụng công thức đó.</p>
</div>

### 5.1.4 Ước lượng posterior

Để fit mô hình vào data ly dị, chúng ta sẽ mở rộng mô hình tuyến tính. Lần nữa, đây là định nghĩa mô hình, với code ở bên phải:

$$ \begin{aligned}
D_i &\sim \text{Normal}( \mu_i, \sigma) \quad && {\Tiny numpyro.sample("D", dist.Normal(mu, sigma), obs=D)}\\
\mu_i &= \alpha + \beta_M M_i + \beta_A A_i \quad && {\Tiny numpyro.deterministic("mu", a + bM * M + bA * A)}\\
\alpha &\sim \text{Normal}(0, 0.2) \quad && {\Tiny a = numpyro.sample("a", dist.Normal(0, 0.2))}\\
\beta_M &\sim \text{Normal}(0, 0.5) \quad && {\Tiny bM = numpyro.sample("bM", dist.Normal(0, 0.5))}\\
\beta_A &\sim \text{Normal}(0, 0.5) \quad && {\Tiny bA = numpyro.sample("bA", dist.Normal(0, 0.5))}\\
\sigma &\sim \text{Exponential}(1) \quad && {\Tiny sigma = numpyro.sample("sigma", dist.Exponential(1))}\\
\end{aligned} $$

Và đây là code `SVI` và `AutoLaplaceApproximation` để ước lượng phân phối posterior:

<b>code 5.10</b>
```python
def model(M, A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM * M + bA * A)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
m5_3 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_3, optim.Adam(1), Trace_ELBO(), M=d.M.values, A=d.A.values, D=d.D.values)
p5_3, losses = svi.run(random.PRNGKey(0), 1000)
post = m5_3.sample_posterior(random.PRNGKey(1), p5_3, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean       std    median      5.5%     94.5%     n_eff     r_hat
    a  -0.00      0.10     -0.01     -0.16      0.14   1049.96      1.00
   bA  -0.61      0.16     -0.61     -0.86     -0.36    822.38      1.00
   bM  -0.06      0.16     -0.06     -0.31      0.19    984.99      1.00
sigma   0.80      0.08      0.79      0.68      0.92    971.25      1.00
</samp>

Trung bình posterior cho tỉ suất kết hôn, `bM`, là gần với zero, với nhiều xác xuất ở hai bên của zero. Trung bình posterior cho độ tuổi kết hôn, `bA`, là không thay đổi. Tốt hơn nữa là vẽ biểu đồ phân phối posterior cho cả ba mô hình, tập trung vào chỉ những tham số slope $\beta_A$ và $\beta_M$:

<b>code 5.11</b>
```python
coeftab = {
    "m5_1": m5_1.sample_posterior(random.PRNGKey(1), p5_1, (1, 1000,)),
    "m5_2": m5_2.sample_posterior(random.PRNGKey(2), p5_2, (1, 1000,)),
    "m5_3": m5_3.sample_posterior(random.PRNGKey(3), p5_3, (1, 1000,)),
}
az.plot_forest(
    list(coeftab.values()),
    model_names=list(coeftab.keys()),
    var_names=["bA", "bM"],
    hdi_prob=0.89,
)
```

![](/assets/images/forest 5-1.svg)

Trung bình posterior được thể hiện bằng dấu chấm và khoảng tin cậy 89% bởi đường nằm ngang. Nhận thấy rằng, `bA` hầu như không thay đổi, chỉ không chắc thêm một ít, trong khi đó, `bM` chỉ liên quan với ly dị khi độ tuổi kết hôn không nằm trong mô hình. Bạn có thể diễn giải những phân phố này như nói rằng:

>Khi ta biết thông tin độ tuổi kết hôn trung vị của một bang nào đó, thì có rất ít hoặc không có giá trị dự đoán khi cũng biết thêm tỉ suất kết hơn của bang đó.

Với ký hiệu kỳ lạ đó, $D \perp\\!\\!\perp M \| A$. Nó kiểm tra gợi ý từ DAG thứ hai trước. Bởi vì DAG thứ nhất không suy ra kết luận này,  nó được loại bỏ.

Chú ý rằng điều này không đồng nghĩa là không có giá trị khi biết thêm tỉ suất kết hôn. Kiên định với DAG trước, nếu bạn không thu thập được dữ liệu độ tuổi kết hôn, thì bạn chắc chắn tìm thấy giá trị khi biết thêm tỉ suất kết hôn. $M$ mang tính dự đoán chứ không phải nhân quả. Giả định nếu không có biến số nhân quả khác bị mất trong mô hình (chi tiết hơn trong chương sau), điều này suy ra không có con đường nhân quả trực tiếp quan trọng từ tỉ suất kết hôn đến tỉ lệ ly dị. Mối liên quan giữa tỉ suất kết hôn và tỉ suất ly dị là giả tạo, tạo nên bởi hiệu ứng của độ tuổi kết hôn lên cả tỉ suất kết hôn và tỉ suất ly dị. Tôi sẽ để đó cho bạn đọc khảo sát mối liên quan giữa độ tuổi kết hôn, $A$, và tỉ suất kết hôn, $M$, để hoàn thành bức tranh.

Nhưng tại sao mô hình `m5_3` đạt được suy luận tỉ suất ly dị không thêm thông tin mới, khi chúng ta đã biết độ tuổi kết hôn? Hãy vẽ vài bức tranh.

<div class="alert alert-dark">
<p><strong>Mô phỏng ví dụ ly dị.</strong> Data ly dị là data thực. Nhưng nó sẽ có ích nếu chúng ta mô phỏng kiểu quan hệ nhân quả này giống như trong DAG: $M \gets A \to D$. Mọi DAG đều có thể được mô phỏng, và mô phỏng này sẽ giúp chúng ta thiết kế mô hình để suy luận đúng quan hệ giữa các biến. Trong trường hợp này. bạn chỉ cần mô phỏng ba biến số:</p>
{% highlight python %}
N = 50  # number of simulated States
age = dist.Normal().sample(random.PRNGKey(0), sample_shape=(N,))  # sim A
mar = dist.Normal(age).sample(random.PRNGKey(1))  # sim A -> M
div = dist.Normal(age).sample(random.PRNGKey(2))  # sim A -> D
{% endhighlight %}
<p>Bây giờ nếu bạn dùng những biến này vào các mô hình <code>m5_1</code>, <code>m5_2</code>, <code>m5_3</code>, bạn sẽ thấy suy luận posterior là như nhau. Chúng ta có thể mô phỏng rằng cả $A$ và $M$ ảnh hưởng $D$: {% highlight python %}div = dist.Normal(age+mar).sample(random.PRNGKey(2)){% endhighlight %}. Trong trường hợp đó, hồi quy ngây thơ của $D$ trên $A$ sẽ khuếch đại ảnh hưởng của $A$, cũng giống như hồi quy ngây thở của $D$ trên $M$ sẽ khuếch đại mức độ quan trọng của $M$. Hồi quy đa biến cũng sẽ giúp bạn giải quyết trong tình huống này. Nhưng diễn giải những ước lượng tham số sẽ luôn luôn phụ thuộc vào những gì bạn tin vào mô hình nhân quả, bởi vì nhiều (rất nhiều) mô hình nhân quả điển hình là kiên định với bất kỳ tập ước lượng tham số. Chúng ta sẽ thảo luận vấn đề này sau, vấn đề <strong>TƯƠNG ĐỒNG MARKOV (MARKOV EQUIVALENCE)</strong>.</p></div>

### 5.1.5 Vẽ biểu đồ cho mô hình đa biến

Hãy tạm thời dừng lại, trước khi tiếp tục. Có rất nhiều bộ phận đang chạy ở đây: ba biến số, vài DAG lạ lùng, và ba mô hình. Nếu bạn cảm thấy bối rối, là bởi vì bạn đang tập trung.

Tốt hơn là chúng ta nên vẽ biểu đồ (visualize) những suy luận từ mô hình. Đồ thị cho phân phối posterior trong mô hình hai biến đơn giản, như trong chương trước, là dễ dàng. Nó có chỉ một biến dự đoán, nên một đồ thị phân tán (scatterplot) là đủ để truyền đạt thông tin. Cho nên trong chương trước chúng tôi dùng đồ thị phân tán cho data. Sau đó chúng ta đặt các đường thẳng và khoảng hồi quy để (1) thể hiện trên biểu đồ mối liên quan giữa biến dự đoán và kết cục và (2) để thấy được khả năng thô của mô hình trong dự đoán từng quan sát cụ thể.

Với hồi quy đa biến, bạn sẽ cần nhiều biểu đồ hơn. Có một văn hoá hay gặp là liệt kê nhiều kỹ thuật đồ thị chỉ để giúp hiểu hồi quy tuyến tính đa biến. Không có kỹ thuật nào trong số đó là phù hợp với mọi công việc, và đa số chúng không tổng quát cho những mô hình hiện đại hơn. Cho nên các tiếp cận tôi dùng ở đây là giúp bạn tính ra những gì bạn cần từ mô hình. Tôi đề ra ba ví dụ về biểu đồ diễn giải:
1. Biểu đồ thặng dư của biến dự đoán (Predictor residual plot): Biểu đồ này thể hiện kết cục so với các giá trị thặng dư từ biến dự đoán. Chúng có ích trong việc hiểu mô hình thống kê, nhưng không gì hơn.
2. Biểu đồ dự đoán posterior (Posterior prediction plot): Biểu đồ này  thể hiện dự đoán của mô hình với data thô, hoặc ngược lại là thể hiện sai số trong dự đoán. Chúng là công cụ đánh giá mức độ fit và khả năng dự đoán. Chúng không phải là công cụ nhân quả.
3. Biểu đồ phản thực (Counterfactual plot): Biểu đồ này thể hiện dự đoán suy ra từ những thí nghiệm tưởng tượng. Đồ thị cho phép bạn  khám phá suy luận nhân quả từ việc kiểm soát một hay nhiều biến khác.

Mỗi một loại biểu đồ đều có lợi và hại riêng, dựa trên bối cảnh và câu hỏi đang quan tâm. Trong phần còn lại, tôi sẽ cho bạn thấy cách để tạo ra những biểu đồ này trong bối cảnh data ly dị.

#### 5.1.5.1 Biểu đồ thặng dư của biến dự đoán (Predictor residual plot)

Thặng dư của một biến dự đoán là sai số dự đoán trung bình khi chúng ta sử dụng những biến còn lại vào mô hình hoá biến dự đoán đang quan tâm. Đây là một khái niệm phức tạp, cho nên chúng ta sẽ đi thẳng vào ví dụ, ở đó nó rõ hơn. Lợi ít của việc tính những thứ này là, khi thể hiện biểu đồ so với kết cục, chúng ta có hồi quy hai biến mà đã đặt điều kiện trên tất cả các biến dự đoán còn lại. Nó chỉ còn lại sự biến thiên mà không được mong đợi qua mô hình trung bình, $\mu$, như là hàm số của các biến dự đoán khác.

Trong mô hình tỉ suất lý dị, chúng ta có hai biến dự đoán: (1) tỉ suất kết hôn $M$ và (2) độ tuổi kết hôn trung vị $A$. Để tính thặng dư biến dự đoán cho một trong hai, chúng ta dùng biến dự đoán còn lại để mô hình hoá nó. Cho nên với tỉ suất kết hôn, đây là mô hình chúng ta cần:

$$ \begin{aligned}
M &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta A_i\\
\alpha &\sim \text{Normal}(0,0.2)\\
\beta &\sim \text{Normal}(0,0.5)\\
\sigma &\sim \text{Exponential}(1)\\
\end{aligned}$$

Giống như trước, $M$ là tỉ suất kết hôn và $A$ là độ tuổi kết hôn trung vị. Chú ý rằng bởi vì chúng ta đã chuẩn hoá cả hai biến, chúng ta đã mong đợi trung bình $\alpha$ là xung quanh zero, như trước. Cho nên tôi đã tái sử dụng prior cũ. Đoạn code này sẽ ước lượng posterior:

<b>code 5.13</b>
```python
def model(A, M=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bAM = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bAM * A)
    numpyro.sample("M", dist.Normal(mu, sigma), obs=M)
m5_4 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_4, optim.Adam(0.1), Trace_ELBO(), A=d.A.values, M=d.M.values)
p5_4, losses = svi.run(random.PRNGKey(0), 1000)
```

Và sau đó chúng ta tính ra **THẶNG DƯ (RESDIUAL)** bằng cách trừ tỉ suất kết hôn quan sát được trong mỗi bang với tỉ suất được dự đoán, dựa vào mô hình trên:

<b>code 5.14</b>
```python
post = m5_4.sample_posterior(random.PRNGKey(1), p5_4, (1000,))
post_pred = Predictive(m5_4.model, post)(random.PRNGKey(2), A=d.A.values)
mu = post_pred["mu"]
mu_mean = jnp.mean(mu, 0)
mu_resid = d.M.values - mu_mean
```

Khi thặng dư là số dương, có nghĩa là tỉ suất quan sát được là dư thừa hơn mong đợi của mô hình, giả định bằng độ tuổi kết hôn trung vị của bang đó. Khi thặng dư là số âm, thì có nghĩa là tỉ suất quan sát được thấp hôn mong đợi của mô hình. Nói một cách đơn giản, bang nào có thặng dư là số dương thì có tỉ suất kết hôn cao với độ tuổi kết hôn trung vị, còn bang có thặng dư âm thì có tỉ suất thấp với độ tuổi kết hôn trung vị. Tốt hơn hết là vẽ biểu đồ thể hiện mối quan hệ giữa hai biến này và các giá trị thặng dư. Trong [**HÌNH 5.4**](#f4), trên trái, tôi thể hiện `m5_4` cùng với những đoạn thẳng cho từng thặng dư. Chú ý rằng những thặng dư đó là sự biến thiên của tỉ suất kết hôn bị sót lại, sau khi lấy ra mối quan hệ tuyến tính thuần tuý giữa hai biến.

<a name="f4"></a>![](/assets/images/fig 5-4.svg)
<details class="fig"><summary>Hình 5.4: Hiểu mô hình đa biến thông qua giá trị thặng dư. Hàng trên thể hiện mỗi biến dự đoán hồi quy trên biến còn lại. Chiều dài của các đoạn thẳng nối giữa giá trị kết cục mong đợi của mô hình, đường hồi quy, với giá trị thực, là <i>thặng dư</i>. Trong hàng dưới, tỉ suất ly dị được hồi quy trên thặng dư ở hàng trên. Dưới trái: sự biến thiên thặng dư trong tỉ suất kết hôn cho thấy có ít quan hệ với tỉ suất ly dị. Dưới phải: Tỉ suất ly dị trên thặng dư độ tuổi kết hôn, cho thấy phần biến thiên còn lại, và biến thiên này liên quan với tỉ suất ly dị.</summary>
{% highlight python %}def model1(A, M=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bAM = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bAM * A)
    numpyro.sample("M", dist.Normal(mu, sigma), obs=M)
def model2(M, A=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bMA = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bMA * M)
    numpyro.sample("M", dist.Normal(mu, sigma), obs=A)
def model1b(rM, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    brM = numpyro.sample("brM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + brM*rM)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
def model2b(rA, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    brA = numpyro.sample("brA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + brA * rA)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
def get_mu_resid(model, pre_var, resid_var):
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide,
              optim.Adam(0.1), Trace_ELBO(),
              **{"A":d.A.values, "M": d.M.values})
    param, _ = svi.run(random.PRNGKey(0), 1000)
    post = guide.sample_posterior(random.PRNGKey(1), param, (1000,))
    mu_mean = jnp.mean(
        Predictive(model, post)(random.PRNGKey(2), **{pre_var: d[pre_var].values})['mu'],
        0)
    mu_resid = d[resid_var].values - mu_mean
    return mu_mean, mu_resid
def get_D_pred(model, resid_varname, resid_values, x_seq):
    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide,
              optim.Adam(0.1), Trace_ELBO(),
              **{resid_varname:resid_values, "D": d.D.values})
    param, _ = svi.run(random.PRNGKey(0), 1000)
    post = guide.sample_posterior(random.PRNGKey(1), param, (1000,))
    mu_pred = Predictive(model, post)(random.PRNGKey(2), **{resid_varname: x_seq})['mu']
    mu_mean = jnp.mean(mu_pred, 0)
    mu_PI = jnp.quantile(mu_pred, jnp.array([0.025, 0.975]), 0)
    return mu_mean, mu_PI
fig, axs = plt.subplots(2,2,figsize=(12,12))
mu_mean1, d['M_resid'] = get_mu_resid(model1, "A", "M")
axs[0,0].scatter(d['A'], d['M'])
axs[0,0].errorbar(d['A'], mu_mean1,
                  yerr=np.stack([np.zeros(len(d['M_resid'])), d['M_resid']]), elinewidth=1)
for name in ['WY','ND','HI','DC','ME']:
    x,y = d.loc[d["Loc"]==name]['A'], d.loc[d["Loc"]==name]['M']
    axs[0,0].annotate(name, (x,y))
axs[0,0].set(xlabel="Độ tuổi kết hôn (chuẩn hoá)", ylabel="Tỉ suất kết hôn (chuẩn hoá)")
mu_mean2, d['A_resid'] = get_mu_resid(model2, "M", "A")
axs[0,1].scatter(d['M'], d['A'])
axs[0,1].errorbar(d['M'], mu_mean2,
                  yerr=np.stack([np.zeros(len(d['A_resid'])), d['A_resid']]), elinewidth=1)
for name in ['HI','DC','ID']:
    x,y = d.loc[d["Loc"]==name]['M'], d.loc[d["Loc"]==name]['A']
    axs[0,1].annotate(name, (x,y))
axs[0,1].set(xlabel="Tỉ suất kết hôn (chuẩn hoá)", ylabel="Độ tuổi kết hôn (chuẩn hoá)")
x_seq3 = jnp.linspace(-1.7, 1.7,100)
mu_mean3, mu_PI3 = get_D_pred(model1b, "rM", d['M_resid'].values, x_seq3)
axs[1,0].scatter(d['M_resid'], d['D'])
axs[1,0].plot(x_seq3, mu_mean3)
axs[1,0].fill_between(x_seq3, mu_PI3[0],mu_PI3[1], alpha=0.5)
axs[1,0].set(xlabel="Thặng dư của biến tỉ suất kết hôn",
             ylabel="Tỉ suất ly dị (chuẩn hoá)",
             ylim=(-2,2))
axs[1,0].vlines(0, -2,2,linestyle='dashed', linewidth=1)
for name in ['WY','ND','HI','DC','ME']:
    x,y = d.loc[d["Loc"]==name]['M_resid'], d.loc[d["Loc"]==name]['D']
    axs[1,0].annotate(name, (x,y))
x_seq4 = jnp.linspace(-1.2, 2.5,100)
mu_mean4, mu_PI4 = get_D_pred(model2b, "rA", d['A_resid'].values, x_seq4)
axs[1,1].scatter(d['A_resid'], d['D'])
axs[1,1].plot(x_seq4, mu_mean4)
axs[1,1].fill_between(x_seq4, mu_PI4[0], mu_PI4[1], alpha=0.5)
axs[1,1].set(xlabel="Thặng dư của biến độ tuổi kết hôn",
             ylabel="Tỉ suất ly dị (chuẩn hoá)",
             ylim=(-2,2))
axs[1,1].vlines(0, -2,2,linestyle='dashed', linewidth=1)
for name in ['HI','DC','ID']:
    x,y = d.loc[d["Loc"]==name]['A_resid'], d.loc[d["Loc"]==name]['D']
    axs[1,1].annotate(name, (x,y))
{% endhighlight %}</details>

Bây giờ để sử dụng những thặng dư này, hãy đưa chúng vào trục hoành và so sánh chúng với kết cục quan tâm thực sự, tỉ suất ly dị. Trong [**HÌNH 5.4**](#f4) (dưới trái), tôi vẽ những thặng dư này với tỉ suất ly dị, nằm trên hồi quy tuyến tính giữa hai biến. Bạn có thể nghĩ biểu đồ này là thể hiện mối quan hệ tuyến tính giữa ly dị và tỉ suất kết hôn, sau khi đặt diều kiện lên độ tuổi kết hôn trung vị. Đường dọc giữa nét đứt chỉ điểm cho tỉ suất kết hôn nằm ngay chính xác trên mong đợi từ độ tuổi kết hôn trung vị. Cho nên bang nào nằm bên phải của đường này có tỉ suất kết hôn cao hơn mong đợi. Bang nằm bên trái đường này có tỉ suất thấp hơn. Tỉ suất ly dị trung bình ở cả hai bên đường thẳng là gần bằng nhau, và cho nên đường hồi quy diễn tả rằng có ít mối quan hệ giữa tỉ suất ly dị và tỉ suất kết hôn.

Quy trình này cũng dùng được với những biến còn lại. Hình trên bên phải trong [**HÌNH 5.4**](#f4) cho thấy hồi quy của $A$ trên $M$ và các thặng dư. Ở hình dưới bên phải, những thặng dư được dùng để dự đoán tỉ suất ly dị. Bang nằm bên phải của đường dọc nét đứt có độ tuổi kết hôn trung vị cao hơn mong đợi, và bang nằm bên trái có độ tuổi két hôn trung vị nhỏ hơn mong đợi. Bây giờ chúng ta nhìn thấy tỉ suất ly dị trung bình ở bên phải là thấp hơn bên trái, như biểu diễn của đường hồi quy. Bang nào trong đó người ta kết hôn trễ hơn mong đợi với một tỉ suất kết hôn cụ thể thì thường ít ly dị hơn.

Vậy mục đích chính của tất cả những thứ này là gì? Nó có một giá trị về mặt khái niệm khi thấy những dự đoán theo mô hình được thể hiện so với kết cục, sau khi trừ đi ảnh hưởng của những biến dự đoán khác. Biểu đồ trong [**HÌNH 5.4**](#f4) làm điều đó. Nhưng quy trình này cũng cho một thông điệp là mô hình hồi quy đo lường mối quan hệ còn lại của mỗi biến dự đoán so với kết cục, sau khi đã biết được những biến dự đoán khác. Để tính ra biểu đồ thặng dư của biến dự đoán, bạn phải thực hiện phép tính này bằng tự bản thân bạn. Trong mô hình đa biến hợp nhất lại, nó tự động xảy ra. Cho dù thế nào, chúng ta nên nhớ điều này, bởi vì hồi quy có thể cư xử một cách bất ngờ. Chúng ta sẽ sớm thấy ví dụ như thế.

Hồi quy tuyến tính thực hiện tất cả những đo lường này cùng một lúc với một mô hình tổng rất cụ thể về các mối quan hệ giữa các biến. Nhưng biến dự đoán có thể liên quan với một biến khác theo cách khác mà không phải phép cộng. Logic cơ bản của điều kiện thống kê là không thay đổi trong trường hợp này, nhưng những chi tiết khác chắc chắn là có, và những biểu đồ thặng dư sẽ giúp ích rất nhiều. May mắn thay là vẫn còn biểu đồ khác để giúp hiểu mô hình. Và chúng ta sẽ học nó tiếp theo.

<div class="alert alert-info">
<p><strong>Thặng dư là tham số, không phải data.</strong> Có một truyền thống, đặc biệt trong sinh học, là lấy thặng dư từ một mô hình như là data tong mô hình khác. Ví dụ, nhà sinh học có thể hồi quy kích thước bộ não trên kích thước cơ thể và sau đó dùng thặng dư kích thước não đó như là data trong mô hình khác. Quy trình này là sai hoàn toàn. Thặng dư là không biết được. Chúng là thàm số, biến số với những giá trị không quan sát được. Nếu xem chúng như là giá trị đã biết thì sẽ làm mất đi tính bất định. Cách làm tốt nhất là cho biến đó vào cùng một mô hình,<sup><a name="r83" href="#83">83</a></sup> và tốt hơn nữa là mô hình được thiết kê dưới hướng dẫn của mô hình nhân quả rõ ràng.</p></div>

#### 5.1.5.2 Biểu đồ dự đoán posterior (Posterior predictive plot)

Việc kiểm tra những dự đoán của mô hình so với data quan sát được là quan trọng. Bạn đã làm việc này ở Chương 3, khi bạn mô phỏng các lần tung quả cầu, trung bình hoá trên posterior, và so sánh kết quả mô phỏng với data quan sát được. Loại kiểm tra này rất hữu ích theo nhiều cách. Còn bây giờ, chúng ta sẽ tập trung vào hai mục đích.

1. Mô hình đã ước lượng chính xác phân phối posterior chưa? Golem có thể gây lỗi, cũng như kỹ sư golem. Lỗi có thể dễ dàng chẩn đoán hơn bằng cách so sánh dự đoán với data thô. Cần thêm vài điểm chú ý, bởi vì không phải tất cả mô hình đều cố gắng tạo ra kết quả giống hệt mẫu quan sát. Nhưng sau cùng, bạn sẽ biết mình mong đợi gì từ một phép ước lượng thành công. Bạn sẽ ví dụ sau (Chương 13).
2. Mô hình thất bại như thế nào? Mô hình là những tưởng tượng hữu ích. Cho nên chúng luôn thất bại ở một điểm nào đó. Đôi khi, mô hình fit chính xác nhưng vẫn kém cho mục đích của chúng ta mà nó phải được loại bỏ. Thông thường hơn, mô hình dự đoán tốt ở một vài khía cạnh, nhưng không tốt ở khía cạnh khác. Bằng việc kiểm tra từng trường hợp cụ thể mà mô hình cho dự đoán kém, bạn có thể có ý tưởng làm sao cải thiện nó. Cái khó ở đây là quá trình này lại là mang tính sáng tạo và dựa dẫm vào kiến thức chuyên môn của người phân tích. Không (chưa) robot nào có thể làm việc đó cho bạn. Nó cũng có nguy cơ khi đuổi theo sai số (noise), một chủ đề chúng ta sẽ tập trung ở các chương sau.

Làm sao để chúng ta tạo một phép kiểm tra dự đoán posterior đơn giản ở ví dụ ly dị? Hãy bắt đầu bằng mô phỏng dự đoán, trung bình hoá trên posterior.

<b>code 5.15</b>
```python
# call predictive without specifying new data
# so it uses original data
post = m5_3.sample_posterior(random.PRNGKey(1), p5_3, (int(1e4),))
post_pred = Predictive(m5_3.model, post)(random.PRNGKey(2), M=d.M.values, A=d.A.values)
mu = post_pred["mu"]
# summarize samples across cases
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
# simulate observations
# again no new data, so uses original data
D_sim = post_pred["D"]
D_PI = jnp.percentile(D_sim, q=(5.5, 94.5), axis=0)
```

Đoạn code trên khá giống với những gì bạn đã thấy, nhưng bây giờ sử dụng data quan sát gốc.

Với mô hình đa biến, có nhiều cách khác nhau để thể hiện những mô phỏng này. Đơn giản nhất là chỉ thể hiện dự đoán so với quan sát được. Đoạn code này sẽ thực hiện điều đó, và sau đó thêm một đường thẳng cho dự đoán hoàn hảo và các đoạn thẳng cho khoảng tin cậy của mỗi dự đoán:

<b>code 5.16</b>
```python
ax = plt.subplot(
    ylim=(float(mu_PI.min()), float(mu_PI.max())),
    xlabel="Observed divorce",
    ylabel="Predicted divorce",
)
plt.plot(d.D, mu_mean, "o")
x = jnp.linspace(mu_PI.min(), mu_PI.max(), 101)
plt.plot(x, x, "--")
for i in range(d.shape[0]):
    plt.plot([d.D[i]] * 2, mu_PI[:, i], "b")
fig = plt.gcf()
```

<a name="f5"></a>![](/assets/images/fig 5-5.svg)
<details class="fig"><summary>Hình 5.5: Biểu đồ dự đoán posterior cho mô hình ly dị đa biến, <code>m5_3</code>. Trục hoành là tỉ suất ly dị quan sát được cho mỗi bang. Trục tung là tỉ suất ly dị dự đoán từ mô hình cho mỗi bang, dựa vào tỉ suất kết hôn và độ tuổi kết hôn trung vị của bang đó. Các đoạn thẳng màu cam là những khoảng tin cậy 89%. Đường chéo là nơi mà dự đoán posterior chính xác như mẫu quan sát.</summary>
{% highlight python %}ax = plt.subplot(
    ylim=(float(mu_PI.min()), float(mu_PI.max())),
    xlabel="Ly dị quan sát",
    ylabel="Ly dị dự đoán",
)
plt.plot(d.D, mu_mean, "o")
x = jnp.linspace(mu_PI.min(), mu_PI.max(), 101)
plt.plot(x, x, "--")
for i in range(d.shape[0]):
    plt.plot([d.D[i]] * 2, mu_PI[:, i], color="C1", linewidth=1)
for i in range(d.shape[0]):
    if d.Loc[i] in ["ID", "UT", "RI", "ME"]:
        ax.annotate(
            d.Loc[i], (d.D[i], mu_mean[i]), xytext=(-25, -5), textcoords="offset pixels"
        ){% endhighlight %}</details>

Biểu đồ được thể hiện ở [**HÌNH 5.5** ](#f5). Rất dễ thấy rằng từ sự mô phỏng này là mô hình dự đoán thấp hơn với những bang có tỉ suất ly dị cao nhưng lại dự đoán cao hơn với nhưng bang có tỉ suất ly dị rất thấp. Đây là bình thường. Đó là công việc của hồi quy - nó rất đa nghi với những giá trị cực, cho nên nó mong đợi hồi quy về trung bình. Nhưng đằng sau hiện tượng hồi quy về trung bình tổng quát, một số bang rất khó chịu với mô hình, nằm ở rất xa từ đường chéo. Tôi đã đánh dấu những điểm như vậy, như Idaho(ID) và Utah(UT), cả hai đều có tỉ suất ly dị thấp hơn nhiều so với mong đợi của mô hình. Cách đơn giản nhất để đánh dấu một vài điểm là dùng `plt.annotate`:

<b>code5.17</b>
```python
for i in range(d.shape[0]):
    if d.Loc[i] in ["ID", "UT", "RI", "ME"]:
        ax.annotate(
            d.Loc[i], (d.D[i], mu_mean[i]), xytext=(-25, -5), textcoords="offset pixels"
        )
fig
```

Có gì lạ ở Idaho và Utah? Cả hai bang này có thành phần rất lớn các thành viên của Giáo hội các Thánh hữu Ngày sau của Chúa Jesus Kitô (Church of Jesus Christ of Latter-day Saints). Thành viên trong giáo hội này ít khi ly dị, cho dù nơi nào họ sống. Điều này cho thấy là nếu có một cái nhìn tinh tế hơn về nhân khẩu học của mỗi bang, nhiều hơn chỉ độ tuổi kết hôn trung vị, sẽ giúp ích hơn.

<div class="alert alert-info">
<p><strong>Thống kê hử, nó có gì tốt?</strong> Người ta thường dùng mô hình thống kê là thực hiện những việc mà mô hình thống kê không làm được. Ví dụ, chúng ta muốn biết hiệu ứng này là thực hay là giả tạo. Thật không may, mô hình chỉ định lượng tính bất định chính xác theo cách mà mô hình hiểu vấn đề. Thông thường đáp án cho câu hỏi thế giới thực về sự thật và nhân quả, phụ thuộc vào thông tin không nằm trong mô hình. Ví dụ, bất kỳ tương quan quan sát được giữa biến kết cục và biến dự đoán có thể bị mất đi hoặc nghịch đảo một khi biến dự đoán khác được thêm vào mô hình. Nhưng nếu chúng ta không suy nghĩ về những biến số đúng, chúng ta sẽ không bao giờ nhận ra điều đó. Cho nên mọi mô hình thống kê đều có điểm yếu và cần phải được đánh giá, cho dù ước lượng và rõ ràng hơn là độ chính xác dự đoán của chúng có tốt cỡ nào. Những cuộc đánh giá và tái duyệt mô hình là phép kiểm thực sự  cho các giả thuyết khoa học. Một giả thuyết đúng sẽ được thông qua và thất bại nhiều "phép kiểm" thống kê trên con đường chấp thuận của nó.</p></div>

<div class="alert alert-dark">
<p><strong>Mô phỏng quan hệ giả tạo.</strong> Một cách để có quan hệ giả tạo (spurious) giữa một biến dự đoán và biến kết cục là biến nhân quả đúng, gọi là $x_{thực}$, ảnh hưởng cả hai biến kết cục, $y$, và một biến giả tạo, $x_{giả}$. Nó khá rối, nhưng, mô phỏng tình huống này sẽ giúp ích nhiều và cho thấy cách data giả tạo xuất hiện và chứng minh cho bạn thấy là hồi quy đa biến có khả năng đáng tin cậy trong việc chỉ điểm cho biến dự đoán đúng, $x_{thực}$. Và đây là một mô phỏng rất cơ bản:</p><b>code 5.18</b>
{% highlight python %}N = 100  # number of cases
# x_real as Gaussian with mean 0 and stddev 1
x_real = dist.Normal().sample(random.PRNGKey(0), (N,))
# x_spur as Gaussian with mean=x_real
x_spur = dist.Normal(x_real).sample(random.PRNGKey(1))
# y as Gaussian with mean=x_real
y = dist.Normal(x_real).sample(random.PRNGKey(2))
# bind all together in data frame
d = pd.DataFrame({"y": y, "x_real": x_real, "x_spur": x_spur}){% endhighlight %}
<p>Bây giờ DataFrame <code>d</code> chứa 100 ca mô phỏng. Bởi vì <code>x_real</code> ảnh hưởng cả <code>y</code> và <code>x_spur</code>, bạn có thể nghĩ <code>x_spur</code> là một kết cục khác của <code>x_real</code>, nhưng chúng ta đã sai lầm khi cho nó là một biến dự đoán tiềm năng của <code>y</code>. Kết quả là, cả $x_{thực}$ và $x_{giả}$ đều có tương quan với $y$. Bạn có thể thấy điều này từ biểu đồ phân tán <code>az.plot_pair</code>. Nhưng khi bạn bao gồm cả hai biến $x$ vào mô hình tuyến tính dự đoán $y$, trung bình posterior cho mối quan hệ giữa $y$ và $x_{giả}$ sẽ gần bằng zero.</p></div>

#### 5.1.5.3 Biểu đồ phản thực (Counterfactual plot)

Một biểu đồ suy luận thứ hai thể hiện gợi ý nhân quả của mô hình. Tôi gọi những biểu đồ này là **PHẢN THỰC (COUNTERFACTUAL)**, bởi vì chúng có thể được tạo ra bằng bất kỳ giá trị nào của biến dự đoán bạn thích, ngay cả những kết hợp không quan sát được như độ tuổi kết hôn trung vị rất cao và tỉ suất kết hôn rất cao. Không có bang nào với đặc tính đó, nhưng trong biểu đồ phản thực, bạn có thể hỏi mô hình cho dự đoán với những bang như vậy, câu hỏi như "Tỉ suất ly dị của Utah sẽ như thế nào, nếu độ tuổi kết hôn trung vị cao hơn?" Khi được sử dụng với mục đích rõ ràng, biểu đồ phản thực sẽ giúp bạn hiểu mô hình hơn, cũng như tạo ra những dự đoán cho can thiệp tưởng tượng và định lượng được kết cục quan sát được sẽ đóng góp như thế nào cho một vài nguyên nhân.

Chú ý rằng từ "phản thực" được tái sử dùng rất nhiều trong thống kê và triết học. Nó ít khi có cùng ý nghĩa khi sử dụng bởi tác giả khác nhau. Ở đây, tôi dùng nó để chỉ điểm cho những phép tính mà sử dụng cấu trúc sơ đồ nhân quả, tiến xa hơn phân phối posterior. Nhưng nó cũng có thể được nhắc đến như những câu hỏi về cả quá khứ và tương lai.

Ứng dụng biểu đồ phản thực đơn giản nhất là cho thấy kết cục thay đổi như thế nào khi bạn thay đổi một biến dự đoán tại một thời điểm. Nếu biến dự đoán $X$ có giá trị mới cho một và nhiều ca trong data của chúng ta, thì kết cục $Y$ sẽ như thế nào? Thay đổi chỉ một biến $X$ cũng có thể thay đổi những biến dự đoán khá, phụ thuộc vào mô hình nhân quả. Giả sử bạn trả cho những cặp tình nhân để họ trì hoãn hôn nhân đến lúc họ 35 tuổi. Chắc chắn điều này sẽ giảm số lượng cặp tình nhân mà kết hôn - vài người trong số họ sẽ chết trước khi 35, trong vô vàn lý do khác - làm giảm tỉ suất kết hôn chung. Một mức độ kiểm soát con người khác thường và tàn nhẫn sẽ cần có để thực sự kiềm chế tỉ suất kết hôn hằng định trong khi ép buộc mọi người kết hôn ở tuổi lớn hơn.

Cho nên hãy xem cách để tạo ra biểu đồ thể hiện dự đoán mô hình mà có thêm yếu tố cấu trúc nhân quả. Công thức cơ bản là:
1. Chọn một biến để kiểm soát, là biến can thiệp.
2. Định nghĩa một khoảng giá trị để đặt biến can thiệp vào.
3. Với mỗi giá trị của biến can thiệp, và với mỗi mẫu trong posterior, sử dụng mô hình nhân quả để mô phỏng giá trị của biến còn lại, bao gồm biến kết cục.

Sau cùng, bạn có được phân phối posterior của kết cục phản thực mà bạn có thể vẽ biểu đồ và tóm tắt theo nhiều cách, dựa vào mục đích của bạn.

Hãy xem lại cách làm này trên mô hình ly dị. Lần nữa chúng ta lấy DAG này:

![](/assets/images/dag 5-1.svg)

Để mô phỏng từ đây, chúng ta cần nhiều hơn DAG. Chúng ta cũng cần một vài lệnh để giúp chúng ta tạo ra các biến. Để đơn giản, chúng ta sẽ dùng phân phối Gaussian cho mỗi biến, giống như mô hình `m5_3`. Nhưng mô hình `m5_3` bỏ quả giả định $A$ ảnh hưởng $M$. Chúng ta thực sự không cần nó để ước lượng $A \to D$. Nhưng chúng ta cần nó để dự đoán hệ quả của việc kiểm soát $A$, bởi vì vài hiệu ứng từ $A$ chạy qua $M$.

Để ước lượng ảnh hưởng của $A$ lên $M$, tất cả những gì chúng ta cần là hồi quy $A$ trên $M$. Không có biến nào khác trong DAG để tạo mối quan hệ giữa $A$ va $M$. Chúng ta chỉ cần thêm hồi quy này vào hàm `model`, chạy hai hồi quy cùng một lúc:

<b>code 5.19</b>
```python
def model(A, M=None, D=None):
    # A -> M
    aM = numpyro.sample("aM", dist.Normal(0, 0.2))
    bAM = numpyro.sample("bAM", dist.Normal(0, 0.5))
    sigma_M = numpyro.sample("sigma_M", dist.Exponential(1))
    mu_M = aM + bAM * A
    M = numpyro.sample("M", dist.Normal(mu_M, sigma_M), obs=M)
    # A -> D <- M
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bM * M + bA * A
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)
m5_3_A = AutoLaplaceApproximation(model)
svi = SVI(
    model, m5_3_A, optim.Adam(0.1), Trace_ELBO(), A=d.A.values, M=d.M.values, D=d.D.values
)
p5_3_A, losses = svi.run(random.PRNGKey(0), 1000)
```

Nhìn vào `print_summary` của posterior của `m5_3_A`. Bạn sẽ thấy $M$ và $A$ có tương quan âm rất mạnh. Nếu chúng ta diễn giải theo nhân quả, thì kiểm soát $A$ sẽ giảm $M$.

Mục tiêu là mô phỏng những gì sẽ xảy ra, nếu chúng ta kiểm soát $A$. Cho nên tiếp theo chúng ta định nghĩa khoảng giá trị của $A$.

<b>code 5.20</b>
```python
A_seq = jnp.linspace(-2, 2, num=30)
```

Code này định nghĩa một danh sách 30 can thiệp tưởng tượng, có khoảng từ 2 độ lệch chuẩn dưới và trên trung bình. Bây giờ chúng ta có thể `Predictive`, như đã gặp ở chương trước, để mô phỏng các quan sát từ mô hình `m5_3_A`. Nhưng lần này chúng ta sẽ nói nó mô phỏng cả $M$ và $D$, theo thứ tự đó. Tại sao theo thứ tự đó? Bởi vì chúng ta phải mô phỏng ảnh hưởng của $A$ lên $M$ trước khi chúng ta mô phỏng ảnh hưởng kết hợp của $A$ và $M$ lên $D$.

<b>code 5.21</b>
```python
# prep data
sim_dat = dict(A=A_seq)

# simulate M and then D, using A_seq
post = m5_3_A.sample_posterior(random.PRNGKey(1), p5_3_A, (1000,))
s = Predictive(m5_3_A.model, post)(random.PRNGKey(2), **sim_dat)
```

Toàn bộ quy trình là như vậy. Nhưng hãy chí ít nhìn vào phần thông tin thêm cuối phần này, ở đó tôi cho bạn thấy các bước cụ thể, để bạn có thể thực hiện mô phỏng phản thực này cho bất kỳ mô hình được fit bằng bất kỳ phần mềm. Bây giờ hãy vẽ dự đoán ra:

<b>code 5.22</b>
```python
plt.plot(sim_dat["A"], jnp.mean(s["D"], 0))
plt.gca().set(ylim=(-2, 2), xlabel="kiểm soát $A$", ylabel="phản thực $D$")
plt.fill_between(
    sim_dat["A"], *jnp.percentile(s["D"], q=(5.5, 94.5), axis=0), color="C1", alpha=0.2
)
plt.title("Toàn bộ hiệu ứng của $A$ lên $D$")
```

<a name="f6"></a>![](/assets/images/fig 5-6.svg)
<details class="fig"><summary>Hình 5.6 Biểu đồ phản thực cho mô hình ly dị đa biến, <code>m5_3</code>. Biểu đồ này thể hiện dự đoán hiệu ứng của việc kiểm soát độ tuổi kết hôn $A$ lên tỉ suất ly dị $D$. Trái: Toàn bộ hiệu ứng nhân quả của việc kiểm soát $A$ (trục hoành) lên $D$. Biểu đồ này chứa hai con đường, $A \to D$ và $A \to M \to D$. Phải: Giá trị $M$ được mô phỏng cho thấy ước lượng của ảnh hưởng $A \to M$.</summary>
{% highlight python %}fig, axs= plt.subplots(1,2,figsize=(12,5))
axs[0].plot(sim_dat["A"], jnp.mean(s["D"], 0))
axs[0].set(ylim=(-2, 2), xlabel="kiểm soát $A$", ylabel="phản thực $D$",
           title="Toàn bộ hiệu ứng của $A$ lên $D$")
axs[0].fill_between(
    sim_dat["A"], *jnp.percentile(s["D"], q=(5.5, 94.5), axis=0), color="C1", alpha=0.2
)
axs[1].plot(sim_dat["A"], jnp.mean(s["M"], 0))
axs[1].set(ylim=(-2, 2), xlabel="kiểm soát $A$", ylabel="phản thực $M$",
           title="Hiệu ứng của $A$ lên $M$")
axs[1].fill_between(
    sim_dat["A"], *jnp.percentile(s["M"], q=(5.5, 94.5), axis=0), color="C1", alpha=0.2
){% endhighlight %}</details>

Kết quả được thể hiện ở [**HÌNH 5.6**](#f6) (bên trái). Nó dự đoán xu hướng của $D$ bao gồm hai con đường: $A \to D$ và $A \to M \to D$. Chúng ta đã trước đó tìm thấy $M \to D$ là rất nhỏ, cho nên đường thứ hai không đóng góp nhiều vào xu hướng. Nhưng nếu $M$ ảnh hưởng mạnh đến $D$, thì đoạn code trên đã bao gôm luôn hiệu ứng. Mô phỏng phản thực này cũng tạo ra các giá trị cho $M$. Nó được thể hiện ở bên trai [**HÌNH 5.6**](#f6). Đối tượng `s` trong code trên bao gồm các giá trị $M$ mô phỏng. Hãy thử tạo lại hình bằng chính bạn.

Dĩ nhiên các phép tính này cũng cho phép tóm tắt bằng các con số. Ví dụ, hiệu ứng nhân quả mong đợi từ việc tăng độ tuổi kết hôn trung vị từ 20 lên 30 là:

<b>code 5.23</b>
```python
# new data frame, standardized to mean 26.1 and stddev 1.24
sim2_dat = dict(A=(jnp.array([20, 30]) - 26.1) / 1.24)
s2 = Predictive(m5_3_A.model, post, return_sites=["M", "D"])(
    random.PRNGKey(2), **sim2_dat
)
jnp.mean(s2["D"][:, 1] - s2["D"][:, 0])
```
<samp>-4.588497</samp>

Hiệu ứng của nó khá lớn là bốn rưỡi độ lệch chuẩn, có lẽ là lớn quá mức.

Mánh khoé của việc mô phỏng phản thực là nhận biết rằng khi chúng ta kiểm soát một biến $X$ nào đó, chúng ta đã phá vỡ ảnh hưởng nhân quả của biến khác lên $X$. Điều này giống như nói rằng chúng ta tuỳ chỉnh DAG để không có mũi tên nào vào $X$. Giả sử chúng ta bây giờ mô phỏng hiệu ứng của việc kiểm soát $M$. Tức là DAG:

![](/assets/images/dag 5-4.svg)

Mũi tên $A \to M$ đã được xoá, bởi vì chúng ta kiểm soát giá trị của $M$, thi $A$ không còn ảnh hưởng lên nó. Đây giống như một thí nghiệm kiểm soát hoàn hảo. Bây giờ chúng ta có thể tuỳ chỉnh đoạn code trên để mô phỏng kết quả phản thực của việc kiểm soát $M$. Chúng ta sẽ mô phỏng tình huống phản thực với $A =0$, và thử xem thay đổi $M$ sẽ như thế nào.

<b>code 5.24</b>
```python
sim_dat = dict(M=jnp.linspace(-2, 2, num=30), A=0)
s = Predictive(m5_3_A.model, post)(random.PRNGKey(2), **sim_dat)["D"]

plt.plot(sim_dat["M"], jnp.mean(s, 0))
plt.gca().set(ylim=(-2, 2), xlabel="kiểm soát $A$", ylabel="phản thực $D$")
plt.fill_between(
    sim_dat["M"], *jnp.percentile(s, q=(5.5, 94.5), axis=0), color="k", alpha=0.2
)
plt.title("Toàn bộ hiệu ứng của $M$ lên $D$")
```

<a name="f7"></a>![](/assets/images/fig 5-7.svg)
<details class="fig"><summary>Hình 5.7: Hiệu ứng phản thực của việc kiểm soát tỉ suất kết hôn $M$ lên tỉ suất ly hôn $D$. Bởi vì $M \to D$ được ước lượng là rất nhỏ, nên không có xu hướng rõ ràng ở đây. Bằng việc kiểm soát $M$, chúng ta phá bỏ ảnh hưởng của $A$ lên $M$, và điều này xoá bỏ quan hệ giữa $M$ và $D$.</summary>
{% highlight python %}sim_dat = dict(M=jnp.linspace(-2, 2, num=30), A=0)
s = Predictive(m5_3_A.model, post)(random.PRNGKey(2), **sim_dat)["D"]
plt.plot(sim_dat["M"], jnp.mean(s, 0))
plt.gca().set(ylim=(-2, 2), xlabel="kiểm soát $A$", ylabel="phản thực $D$")
plt.fill_between(
    sim_dat["M"], *jnp.percentile(s, q=(5.5, 94.5), axis=0), color="C1", alpha=0.2
)
plt.title("Toàn bộ hiệu ứng của $M$ lên $D$"){% endhighlight %}</details>

Bây giờ chúng ta chỉ mô phỏng $D$. Chúng ta không mô phỏng $A$, bởi vì $M$ không ảnh hưởng nó nữa. Tôi thể hiện biểu đồ này ở [**HÌNH 5.7**](#f7). Xu hướng này không mạnh, bởi vì không có bằng chứng cho ảnh hưởng mạnh của $M$ lên $D$.

Ở mô hình phức tạp hơn với nhiều con đường tiềm năng, chiến thuật tương tự vẫn sẽ tính hiệu ứng phản thực của một yếu tố phơi nhiễm. Nhưng như bạn sẽ thấy ở ví dụ sau, thông thường nó không đủ khả năng để ước lượng hiệu ứng nhân quả phù hợp, không bị nhiễu của vài yếu tố phơi nhiễm $X$ lên kết cục $Y$. Cho nên chúng ta sẽ trở về chủ đề này trong các chương tương lai.

<div class="alert alert-dark">
<p><strong>Mô phỏng phản thực.</strong> Ví dụ trong phần này sử dụng <code>Predictive</code> để giấu đi các chi tiết. Nhưng tự mình thực hiện mô phỏng hiệu ứng phản thực là không khó. Nó chỉ dùng định nghĩa của mô hình. Giả sử chúng ta đã fit mô hình <code>m5_3_A</code>, mô hình bao gồm cả con đường nhân quả $A \to D$ và $A \to M \to D$. Chúng ta định nghĩa một khoảng giá trị mà chúng ta muốn gán vào $A$:</p>
<b>code 5.25</b>
{% highlight python %}A_seq = jnp.linspace(-2, 2, num=30){% endhighlight %}
<p>Tiếp theo chúng ta cần trích xuất mẫu posterior, bởi vì chúng ta sẽ mô phỏng quan sát cho mỗi tập mẫu. Sau đó nó chỉ là vấn đề dùng định nghĩa mô hình với mẫu, trong như các ví dụ trước. Mô hình định nghĩa phân phối của $M$. Chúng ta chỉ chuyển đổi định nghĩa đó thành hàm mô phỏng tương ứng, tức là <code>dist.Normal</code>:</p>
<b>code 5.26</b>
{% highlight python %}post = m5_3_A.sample_posterior(random.PRNGKey(1), p5_3_A, (1000,))
post = {k: v[..., None] for k, v in post.items()}
M_sim = dist.Normal(post["aM"] + post["bAM"] * A_seq).sample(random.PRNGKey(1)){% endhighlight %}
<p>Mô hình tuyến tính nằm trong <code>dist.Normal</code> nằm ngay trong định nghĩa mô hình. Nó tạo ra một ma trận các giá trị, với mẫu theo từng hàng với các ca tương ứng với giá trị của <code>A_seq</code> theo từng hàng. Chúng ta bây giờ có thể có giá trị mô phỏng cho $M$, chúng ta cũng có thể mô phỏng $D$:</p>
<b>code 5.27</b>
{% highlight python %}D_sim = dist.Normal(post["a"] + post["bA"] * A_seq + post["bM"] * M_sim).sample(
    random.PRNGKey(1)
){% endhighlight %}
<p>Nếu bạn vẽ <code>A_seq</code> so với trung bình các cột của <code>D_sim</code>, bạn có thể thấy cùng kết quả như trước. Trong mô hình phức tạp, có thể có nhiều biến để mô phỏng. Nhưng quy trình cơ bản là như nhau.</p></div>

## <center>5.2 Tương quan bị ẩn</center><a name="a2"></a>

Ví dụ tỉ suất ly dị đã chúng minh sử dụng đa biến dự đoán là hữu ích cho việc loại bỏ quan hệ giả tạo. Một lý do thứ hai của việc sử dụng nhiều hơn một biến dự đoán là đo lường ảnh hưởng trực tiếp của nhiều yếu tố đến kết cục, khi mà không có cái nào ảnh hưởng rõ rằng từ mối quan hệ hai biến. Loại vấn đề xảy ra khi có hai biến dự đoán tương quan với nhau. Tuy nhiên, một trong hai biến tương quan dương với kết cục và biến còn lại tương quan âm với nó.

Bạn sẽ xem xét loại vấn đề này ở bối cảnh data mới, thông tin về các thành phần của sữa trong các loài khỉ, cũng như một số thông số về các loài đó, như trọng lượng cơ thể và kích thước não bộ.<sup><a name="r84" href="#84">84</a></sup> Sữa là một đầu tư lớn, còn đắt hơn cả việc mang thai. Tài nguyên đắt tiền này thường được thay đổi theo nhiều cách không rõ ràng, phụ thuộc vào sinh lý và chi tiết phát triển của từng loài hữu nhũ. Hãy tải data về:

<b>code 5.28</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv?raw=true", sep=";")
```

Bạn sẽ thấy trong cấu trúc của DataFrame có 29 dòng cho 8 biến. Các biến mà chúng ta quan tâm ở đây là `kcal.per.g` (kilicalory năng lượng cho mỗi gram sữa), `mass` (trọng lượng cơ thể trung bình ở giống cái, theo kilogram), và `neocortex.perc` (tỉ lệ trọng lượng vỏ não trên trọng lượng toàn phần não).

Một giả thuyết phổ biến là động vật có não lớn hơn thì cho sữa nhiều năng lượng hơn, để não bộ phát triển tốt. Để khẳng định giả thuyết này cần rất nhiều cố gắng trong ngành sinh học tiến hoá, bởi vì còn nhiều vấn đề thống kê khó khăn trong so sánh các loài vật. Nhiều nhà sinh học không có mô hình tham khảo ngoài trừ các công cụ hồi quy, và kết quả của hồi quy thì không thực sự diễn giải được. Ý nghĩa nhân quả của ước lượng thống kê luôn phụ thuộc vào thông tin ngoài data.

Chúng ta không giải quyết câu hỏi đó ở đây. Nhưng chúng ta sẽ khám phá một ví dụ thú vị. Câu hỏi đặt ra là với mức độ năng lượng của sữa, đơn vị là kilocalory, liên quan đến tỉ lệ trọng lượng vỏ não như thế nào. Vỏ não là phần chất xám ngoài cùng của bộ não, mà đặc biệt phát triển ở các loài khỉ. Chúng ta cũng cần trọng lượng giống cái, để xem xét hiện lượng ẩn đi các quan hệ giữa các biến. Hãy chuẩn hoá ba biến này. Cũng giống như ví dụ trước, chuẩn hoá giúp ước lượng posterior đáng tin cậy hơn cũng như xây dụng prior hợp lý.

<b>code 5.29</b>
```python
d["K"] = d["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
d["N"] = d["neocortex.perc"].pipe(lambda x: (x - x.mean()) / x.std())
d["M"] = d.mass.map(jnp.log).pipe(lambda x: (x - x.mean()) / x.std())
```

Mô hình đầu tiên là hồi quy đơn giản hai biến giữa kilocalory và tỉ lệ vỏ não. Bạn đã biết cách xây dụng hồi quy này. Ở dạng toán học:

$$ \begin{aligned}
K_i   &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &=    \alpha + \beta_N N_i\\
\end{aligned} $$

Trong đó $K$ là kilocalory dược chuẩn hoá và $N$ là tỉ lệ vo não được chuẩn hoá. Chúng ta cần phải chọn các prior. Nhưng trước tiên hãy chạy thử `SVI` với prior mơ hồ, bởi vì còn một vấn đề thiết kế chính cần giải quyết trước.

<b>code 5.30</b>
```python
def model(N, K):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bN = numpyro.sample("bN", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bN * N
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
with numpyro.validation_enabled():
    try:
        m5_5_draft = AutoLaplaceApproximation(model)
        svi = SVI(model, m5_5_draft, optim.Adam(1), Trace_ELBO(), N=d.N.values, K=d.K.values)
        p5_5_draft, losses = svi.run(random.PRNGKey(0), 1000)
    except ValueError as e:
        print(str(e))
```

Khi bạn thực thi đoạn code này, bạn sẽ nhận được lỗi: <samp>Normal distribution got invalid loc parameter.</samp>

Chuyện gì đã xảy ra? Lỗi này nghĩa là mô hình không lấy xác suất đúng có các giá trị tham số bắt đầu. Trong trường hợp này, thủ phạm là các giá trị trống ở biến `N`. Nhìn vào trong biến gốc và bạn sẽ thấy:

<b>code 5.31</b>
```python
d["neocortex.perc"]
```
<samp>0     55.16
1       NaN
2       NaN
3       NaN
4       NaN
5     64.54
6     64.54
7     67.64
8       NaN
9     68.85
10    58.85
11    61.69
12    60.32
13      NaN
14      NaN
15    69.97
16      NaN
17    70.41
18      NaN
19    73.40
20      NaN
21    67.53
22      NaN
23    71.26
24    72.60
25      NaN
26    70.24
27    76.30
28    75.49
Name: neocortex.perc, dtype: float64</samp>

Mỗi `NaN` trong kết quả là giá trị mất. Nếu bạn cho vào một vector như vậy vào hàm likelihood như `dist.Normal`, nó không biết làm gì tiếp theo. Suy cho cùng, xác suất của một giá trị mất là gì? Cho dù đáp án thế nào, nó không phải con số, và `dist.Normal` cũng trả về một `NaN`. Không các nào bắt đầu, `SVI` chịu thua và đưa ra lỗi.

Sửa lỗi này thì dễ. Bạn chỉ cần làm ở đây là loại bỏ thủ công các trường hợp chứa `NaN`. Hay còn gọi là **PHÂN TÍCH TRƯỜNG HỢP ĐẦY ĐỦ (COMPLETE CASE ANALYSIS)**. Một số lệnh fit mô hình tự động, sẽ loại bỏ những trường hợp này một cách tự động. Nhưng đó không bao giờ là điều tốt. Đầu tiên, sự hợp lệ của nó phụ thuộc vào trình xử lý gây ra những giá trị này bị mất đi. Trong Chương 15, bạn sẽ khám phá sâu hơn. Thứ hai, một khi bạn bắt đầu so sánh mô hình, bạn phải so sánh những mô hình fit cùng một data. Nếu có vài biến số có giá trị mất mà cái khác không có, những công cụ tự động sẽ tạo ra phép so sánh gây hiểu lầm một cách im lặng.

Giờ chúng ta hãy đi tiếp, loại bỏ các trường hợp có giá trị mất. Bạn nên tốt hơn tự thực hiện nó. Để tạo ra DataFrame chỉ có trường hợp đầy đủ, dùng:

<b>code 5.32</b>
```python
d = d.iloc[d[["K", "N", "M"]].dropna(how="any", axis=0).index]
```

Nó tạo ra DataFrame mới, `dcc`, gồm 17 dòng từ `d` mà không có giá trị mất trong trong ba biến được đưa ra. Bây giờ hãy làm việc với DataFrame mới. Tất cả những gì mới là dùng `dcc` thay cho `d`:

<b>code 5.33</b>
```python
def model(N, K=None):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bN = numpyro.sample("bN", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bN * N)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_5_draft = AutoLaplaceApproximation(model)
svi = SVI(model, m5_5_draft, optim.Adam(0.1), Trace_ELBO(), N=dcc.N.values, K=dcc.K.values)
p5_5_draft, losses = svi.run(random.PRNGKey(0), 1000)
```

Trước khi xem xét dự đoán posterior, hãy xem xét prior. Cũng giống như nhiều bài toán hồi quy tuyến tinh đơn giản khác, prior này là vô hại. Nhưng nó có hợp lý không? Điều quan trọng là phải xây dụng prior hợp lý, bởi vì khi mô hình ít đơn giản hơn, prior có thể giúp ích, nhưng chỉ khi chúng là phù hợp với khoa học. Để mô phỏng và vẽ 50 đường hồi quy prior:

<b>code 5.34</b>
```python
xseq = jnp.array([-2, 2])
prior_pred = Predictive(model, num_samples=1000)(random.PRNGKey(1), N=xseq)
mu = prior_pred["mu"]
plt.subplot(xlim=xseq, ylim=xseq)
for i in range(50):
    plt.plot(xseq, mu[i], "k", alpha=0.3)
```

<a name="f8"></a>![](/assets/images/fig 5-8.svg)
<details class="fig"><summary>Hình 5.8: Phân phối dự đoán prior cho mô hình sữa các loài khỉ đầu tiên, <code>m5_5</code>. Mỗi biểu đồ thể hiện khoảng 2 độ lệch chuẩn cho mỗi biến. Trái: Những dự đoán mơ hồ. Những prior này rõ ràng là ngu ngốc. Phải: Prior ít ngu ngốc hơn nhưng ít nhất nằm trong khoảng các quan sát phù hợp.</summary>
{% highlight python %}def model1(N, K=None):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bN = numpyro.sample("bN", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bN * N)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
def model2(N, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bN = numpyro.sample("bN", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bN * N)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
fig, axs= plt.subplots(1,2,figsize=(10,5), sharey=True)
xseq = jnp.array([-2, 2])
for ax, model in zip(axs, [model1, model2]):
    prior_pred = Predictive(model, num_samples=1000)(random.PRNGKey(1), N=xseq)
    mu = prior_pred["mu"]
    for i in range(50):
        ax.plot(xseq, mu[i], color="C0", alpha=0.3, linewidth=2)
axs[0].set(title="a ~ Normal(0,1)\nbN ~ Normal(0,1)",
           ylim=(-2,2), yticks=[-2,-1,0,1,2],
           ylabel="kilocal mỗi g (chuẩn hoá)",
           xlabel="tỉ lệ vỏ não (chuẩn hoá)")
axs[1].set(title="a ~ Normal(0,0.2)\nbN ~ Normal(0,0.5)", xlabel="tỉ lệ vỏ não (chuẩn hoá)")
plt.tight_layout(){% endhighlight %}</details>

Kết quả được thể hiện ở bên trái [**HÌNH 5.8**](#f8). Tôi đã thể hiện khoảng 2 độ lệch chuẩn cho hai biến. Để nó là khoảng nhiều nhất của không gian kết cục. Những đường này là điên rồ. Như ví dụ trước, chúng ta có thể làm tốt hơn bằng cả việc thắt chặt prior $\alpha$ để nó gần zero hơn. Với hai biến số được chuẩn hoá, khi biến dự đoán là zero, thì giá trị mong đợi của kết cục sẽ là zero. Và slope $\beta_N$ cũng cần thắt chặt, để nó không thường xuyên tạo ra nhưng quan hệ mạnh không phù hợp. Đây là mô hình của chúng ta:

<b>code 5.35</b>
```python
def model(N, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bN = numpyro.sample("bN", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bN * N)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_5 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_5, optim.Adam(1), Trace_ELBO(), N=dcc.N.values, K=dcc.K.values)
p5_5, losses = svi.run(random.PRNGKey(0), 1000)
```

Nếu bạn vẽ biểu đồ cho những prior này, bạn sẽ nhận được giống như bên phải của [**HÌNH 5.8**](#f8). Chúng vẫn là những prior mơ hồ, nhưng ít nhất các đường thẳng nằm trong khoảng xác suất cao của data quan sát được.

Hãy nhìn vào tóm tắt posterior:

<b>code 5.36</b>
```python
post = m5_5.sample_posterior(random.PRNGKey(1), p5_5, (1000,))
print_summary({x: post[x] for x in ['a','bN','sigma']}, 0.89, False)
```
<samp>        mean       std    median      5.5%     94.5%     n_eff     r_hat
    a   0.05      0.16      0.05     -0.21      0.29    931.50      1.00
   bN   0.13      0.23      0.13     -0.21      0.53   1111.88      1.00
sigma   1.05      0.18      1.03      0.78      1.35    944.03      1.00</samp>

Từ tóm tắt này, bạn có thể thấy rằng nó không phải là tương quan mạnh hay vô cùng chính xác. Độ lệch chuẩn hầu như gấp đôi trung bình của nó. Cũng như mọi khi, sẽ dễ hơn nếu chúng ta vẽ một bức tranh. Các bảng còn số là giọng nói của golem, và chúng ta không phải golem. Chúng ta có thể vẽ dự đoán trung bình và khoảng tin cậy 89% của nó để thấy rõ hơn. Code dưới đây không có gì lạ. Chỉ có khoảng giá $N$ lớn hơn, trong `xseq`, để biểu đồ nhìn đẹp hơn.

<b>code 5.37</b>
```python
xseq = jnp.linspace(start=dcc.N.min() - 0.15, stop=dcc.N.max() + 0.15, num=30)
post = m5_5.sample_posterior(random.PRNGKey(1), p5_5, (1000,))
post_pred = Predictive(m5_5.model, post)(random.PRNGKey(2), N=xseq)
mu = post_pred["mu"]
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
az.plot_pair(dcc[["N", "K"]].to_dict(orient="list"))
plt.plot(xseq, mu_mean, "k")
plt.fill_between(xseq, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
```

<a name="f9"></a>![](/assets/images/fig 5-9.svg)
<details class="fig"><summary>Hình 5.9: Năng lượng sữa và vỏ não giữa các loài khỉ. Ở hai biểu đồ trên, hồi quy hai biến đơn giản của kilocalory mỗi gram sữa (K) lên (trái) tỉ lệ vỏ não (N) và (phải) logarith thể trọng giống cái (M) cho thấy quan hệ yếu. Ở hàng dưới, mô hình với cả tỉ lệ vỏ não (N) và log thể trọng (M) cho thấy quan hệ mạnh hơn.</summary>
{% highlight python %}def model(M, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM*M)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_6 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_6, optim.Adam(1), Trace_ELBO(), M=dcc.M.values.astype("float"), K=dcc.K.values)
p5_6, losses = svi.run(random.PRNGKey(0), 1000)
def model(M,N,K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bN = numpyro.sample("bN", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM*M + bN*N)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_7 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_7, optim.Adam(1), Trace_ELBO(),
          M=dcc.M.values.astype("float"), N=dcc.N.values, K=dcc.K.values)
p5_7, losses = svi.run(random.PRNGKey(0), 1000)
fig, axs = plt.subplots(2,2, figsize=(10,8), sharey=True, sharex=True)
for ax, m, p ,k in zip(axs[0], [m5_5,m5_6],[p5_5,p5_6], ['N','M']) :
    xseq = jnp.linspace(start=dcc[k].min() - 0.15, stop=dcc[k].max() + 0.15, num=30)
    post = m.sample_posterior(random.PRNGKey(1), p, (1000,))
    post_pred = Predictive(m.model, post)(random.PRNGKey(2), **{k:xseq})
    mu = post_pred["mu"]
    mu_mean = jnp.mean(mu, 0)
    mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
    az.plot_pair(dcc[[k, "K"]].to_dict(orient="list"), ax=ax)
    ax.plot(xseq, mu_mean, "C0")
    ax.fill_between(xseq, mu_PI[0], mu_PI[1], color="C1", alpha=0.2)
for ax, k, k1 in zip(axs[1], ['N','M'], ['M','N']):
    xseq = jnp.linspace(start=dcc[k].min() - 0.15, stop=dcc[k].max() + 0.15, num=30)
    post = m5_7.sample_posterior(random.PRNGKey(1), p5_7, (1000,))
    post_pred = Predictive(m5_7.model, post)(random.PRNGKey(2), **{k:xseq, k1:0})
    mu = post_pred["mu"]
    mu_mean = jnp.mean(mu, 0)
    mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
    ax.plot(xseq, mu_mean, "C0")
    ax.fill_between(xseq, mu_PI[0], mu_PI[1], color="C1", alpha=0.2)
    ax.set(title=f"Phản thực với {k1}=0")
for ax in axs[:,0]:
    ax.set(xlabel="tỉ lệ vỏ não (chuẩn hoá)", ylabel="kilocal mỗi g (chuẩn hoá)")
for ax, k in zip(axs[:,1], ['M','N']):
    ax.set(xlabel="logarith thể trọng (chuẩn hoá)", ylabel="kilocal mỗi g (chuẩn hoá)")
plt.tight_layout()
line = plt.Line2D([0,1], [0.53,0.53],color='k',linewidth=1, alpha=0.6)
fig.add_artist(line){% endhighlight %}</details>

Tôi thể hiện biểu đồ này ở hình trên bên trái của [**HÌNH 5.9**](#f9). Đường trung bình posterior có tương quan dương yếu, nhưng nó rất không chính xác. Rất nhiều slope dương và âm đều phù hợp, dưới mô hình và data này.

Giờ hãy xem xét biến dự đoán khác, thể trọng giống cái, `mass` trong DataFrame. Hãy sử dụng logarith của thể trọng, `jnp.log(d['mass'])`, làm biến dự đoán. Tại sao logarith của thể trọng thay vì thể trọng thô ở kilogram? Thông thường thì các giá trị đo lường như thể trọng liên quan đến các biến khác thông qua mức độ (magnitude). Lấy logarith của giá trị đo lường biến chúng thành các mức độ. Cho nên bằng cách sử dụng logarith của thể trọng này, chúng ta đang nói rằng là đang nghi vấn mức độ của thể trọng giống cái liên quan đến năng lượng trong sữa, theo phong cách tuyến tính. Sau này, trong Chương 16, bạn sẽ gặp tại sao quan hệ logarith này là kết quả không thể tránh được do nguyên nhân vật lý học của sinh vật.

Bây giờ chúng ta dựng một mô hình tương tự, những xem xét quan hệ hai biến giữa kilocalory và thể trọng. Bởi vị thể trọng được chuẩn hoá, chúng ta có thể dùng prior giống như trên và vẫn ở trong khoảng kết cục phù hợp. Nhưng nếu bạn là một chuyên gia trong lĩnh vực phát triển, bạn có thể làm khá hơn điều này.

<b>code 5.38</b>
```python
def model(M, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM * M)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_6 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_6, optim.Adam(1), Trace_ELBO(), M=dcc.M.values, K=dcc.K.values)
p5_6, losses = svi.run(random.PRNGKey(0), 1000)
post = m5_6.sample_posterior(random.PRNGKey(1), p5_6, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean       std    median      5.5%     94.5%     n_eff     r_hat
    a   0.06      0.16      0.06     -0.20      0.29    931.50      1.00
   bM  -0.28      0.20     -0.28     -0.61      0.03   1088.44      1.00
sigma   0.99      0.17      0.98      0.72      1.26    957.10      1.00</samp>

Log-thể trọng là tương quan âm với kilocalory. Quan hệ này có vẻ mạnh hơn so với tỉ lệ vỏ não, mặc dù theo hương ngược lại. Nhưng nó vấn không chắc chắn, với khoảng tin cậy rộng, kiên định với khoảng quan hệ yếu và mạnh. Hồi quy này được thể hiện với biểu đồ trên bên phải của [**HÌNH 5.9**](#f9). Bạn nên tuỳ biến code của biểu đồ trên bên trái trong cùng hình trên, để chắc rằng bạn hiểu cách làm như thế nào.

Bây giờ hãy xem chuyện gì xảy ra nếu chúng ta thêm cả hai biến dự đoán cùng lúc vào hồi quy. Đây là mô hình đa biến, ở dạng toán học:

$$ \begin{aligned}
K_i     &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i   &= \alpha + \beta_N N_i + \beta_M M_i\\
\alpha  &\sim \text{Normal}(0, 0.2)\\
\beta_N &\sim \text{Normal}(0, 0.5)\\
\beta_M &\sim \text{Normal}(0, 0.5)\\
\sigma  &\sim \text{Exponential}(1)\\
\end{aligned} $$

Ước lượng posterior không cần thêm code mới:

<b>code 5.39</b>
```python
def model(N, M, K=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bN = numpyro.sample("bN", dist.Normal(0, 0.5))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bN * N + bM * M)
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m5_7 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m5_7, optim.Adam(1), Trace_ELBO(), N=dcc.N.values, M=dcc.M.values, K=dcc.K.values
)
p5_7, losses = svi.run(random.PRNGKey(0), 1000)
post = m5_7.sample_posterior(random.PRNGKey(1), p5_7, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean       std    median      5.5%     94.5%     n_eff     r_hat
    a   0.06      0.13      0.06     -0.15      0.26   1049.96      1.00
   bM  -0.68      0.23     -0.68     -1.06     -0.32    837.54      1.00
   bN   0.65      0.25      0.66      0.25      1.06    885.39      1.00
sigma   0.77      0.14      0.77      0.55      0.97   1029.58      1.00</samp>

Bằng cách thêm cả hai biến dự đoán vào hồi quy, quan hệ posterior của cả hai với kết cục đều tăng lên. Biểu diễn trên hình vẽ để so sánh posterior này với những posterior trước sẽ giúp nhìn rõ hơn sự thay đổi này:

<b>code 5.40</b>
```python
coeftab = {
    "m5.5": m5_5.sample_posterior(random.PRNGKey(1), p5_5, (1, 1000,)),
    "m5.6": m5_6.sample_posterior(random.PRNGKey(2), p5_6, (1, 1000,)),
    "m5.7": m5_7.sample_posterior(random.PRNGKey(3), p5_7, (1, 1000,)),
}
az.plot_forest(
    list(coeftab.values()),
    model_names=list(coeftab.keys()),
    var_names=["bM", "bN"],
    hdi_prob=0.89,
)
```

![](/assets/images/forest 5-2.svg)

Trung bình posterior cho tỉ lệ vỏ não và log-thể trọng đều đã rời xa ra zero. Thêm cả hai biến dự đoán vào mô hình có vẻ đã làm cho ước lượng của chúng tách xa nhau.

Chuyện gì đã xảy ra? Tại sao thêm tỉ lệ vỏ não và thể trọng vào cùng mô hình dẫn đến quan hệ mạnh cho cả hai? Đây là một tình huống mà trong đó hai biến tương quan với kết cục, nhưng một biến thì tương quan dương với nó và biến còn lại thì tương quan âm với nó. Thêm nữa, cả hai biến giải thích đều tương quan dương với nhau. Hãy thử `az.plot_pair(dcc[['K','M','N']].to_dict(orient='list'))` để thấy được dạng quan hệ này. Kết quả của dạng quan hệ này là các biến có xu hướng loại trừ một biến khác.

Đây là một trường hợp khác trong đó hồi quy đa biến tự động tìm ra trường hợp rõ ràng nhất và dùng chúng tạo ra suy luận. Những gì mô hình hồi quy làm là hỏi rằng nếu loài có tỉ lệ vỏ não nhiều *với thể trọng của nó* có năng lượng sữa cao hay không. Tương tự, mô hình hỏi rằng nếu loài có thể trọng lớn *với tỉ lệ vỏ não của nó* có năng lượng sữa cao hay không. Loài lớn hơn, như tinh tinh, có sữa ít năng lượng. Nhưng loài với nhiều vỏ não hơn thì sữa giàu năng lượng lượng hơn. Sự thật mà hai biến này, thể trọng và vỏ não, tương quan giữa các loài làm cho nó khó nhìn ra những quan hệ này, trừ phi chúng ta quan tâm đến cả hai.

Một vài DAG sẽ giúp đỡ chúng ta. Có ít nhất ba sơ đồ kiên định với data này.

![](/assets/images/dag 5-5.svg)

Từ bên trái, khả năng đầu tiên là thể trọng ($M$) ảnh hưởng lên tỉ lệ vỏ não ($N$). Cả hai sau đều ảnh hưởng lên kilocalory trong sữa ($K$). Thứ hai, ở giữa, vỏ não có thể thay vào đó ảnh hưởng lên thể trọng, Hai biến đều tương quan với nhau như trong mẫu. Cuối cùng, bên phải, có thể có một biến không quan sát được là $U$ ảnh hưởng lên cả $M$ và $N$, tạo ra tương quan giữa chúng. Trong sách này, tôi sẽ đánh dấu biến không quan sát được bằng dấu chấm. Một mối nguy hiểm trong suy luận nhân quả là nó có thể có nhiều biến không quan sát được có thể ảnh hưởng lên kết cục hoặc biến dự đoán. Chúng ta sẽ xem xét vấn đề này nhiều hơn ở chương sau.

Vậy DAG nào ở trên là đúng? Ta không thể kiểm chứng được chỉ với data, bởi vì những DAG này đều có chung một tập **MỐI QUAN HỆ ĐỘC LẬP CÓ ĐIỀU KIỆN (CONDITIONAL INDEPENDENCIES)**. Trong trường hợp này, không có mối quan hệ độc lập có điều kiện nào - mỗi DAG trên đều suy ra là tất cả các cặp biến đều có quan hệ với nhau, cho dù chúng ta đặt điều kiện trên đâu. Một tập hợp DAG có cùng tập mối quan hệ có điều kiện được gọi là một tập **TƯƠNG ĐỒNG MARKOV (MARKOV EQUIVALENCE)**. Trong phần thông tin thêm, tôi sẽ chỉ bạn cách mô phỏng các quan sát kiên định với những DAG này, cách mà chúng có thể tạo ra hiện tương bị ẩn này, và cách sử dụng `causalgraphicalmodels` để tìm ra các DAG trong một tập tương đồng Markov. Nhớ rằng trong khi chỉ data đơn thuần không bao giờ nói bạn biết mô hình nhân quả nào là đúng, kiến thức khoa học của bạn về các biến sẽ loại trừ một lượng lớn các DAG ngu ngốc, nhưng tương đồng Markov.

Việc sau cùng chúng ta muốn làm với mô hình này là hoàn thành [**HÌNH 5.9**](#f9). Hãy vẽ biểu đồ phản thực lần nữa. Giả sử DAG thứ ba là đúng. Sau đó tưởng tượng kiểm soát biến $M$ và $N$, phá bỏ ảnh hưởng của $U$ lên chúng. Trong thực tế, thí nghiệm này là không khả thi. Nếu chúng ta đổi kích thước con vật, thì chọn lọc tự nhiên cũng sẽ thay đổi những đặc tính khác để phù hợp với nó. Nhưng biểu đồ phản thực sẽ giúp chúng ta thấy được cách nhìn của mô hình về liên quan giữa mỗi biến dự đoán và biến kết cục. Đây là code để tạo ra biểu đồ dưới bên trái trong [**HÌNH 5.9**](#f9).

<b>code 5.41</b>
```python
xseq = jnp.linspace(start=dcc.N.min() - 0.15, stop=dcc.N.max() + 0.15, num=30)
post = m5_7.sample_posterior(random.PRNGKey(1), p5_7, (1000,))
post_pred = Predictive(m5_7.model, post)(random.PRNGKey(2), M=0, N=xseq)
mu = post_pred["mu"]
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)
plt.subplot(xlim=(dcc.M.min(), dcc.M.max()), ylim=(dcc.K.min(), dcc.K.max()))
plt.plot(xseq, mu_mean, "C0")
plt.fill_between(xseq, mu_PI[0], mu_PI[1], color="C1", alpha=0.2)
```

Bạn nên thử tạo lại biểu đồ dưới bên phải bằng tuỳ biến code này. Trong phần thực hành, tôi sẽ yêu cầu bạn xem xét điều gì sẽ xảy ra, nếu chúng ta chọn một trong những DAG khác.

<div class="alert alert-dark">
<p><strong>Mô phỏng quan hệ bị ẩn.</strong> Cũng giống như để hiểu tương quan giả tạo, việc mô phỏng data trong đó hai biến dự đoán che đậy nhau, sẽ giúp ích. Trong phần trước, tôi hiển thị ba DAG kiên định với điều này. Để mô phỏng data kiên định với DAG đầu tiên:</p>

<b>code 5.42</b>
{% highlight python %}# M -> K <- N
# M -> N
n = 100
M = dist.Normal().sample(random.PRNGKey(0), (n,))
N = dist.Normal(M).sample(random.PRNGKey(1))
K = dist.Normal(N - M).sample(random.PRNGKey(2))
d_sim = pd.DataFrame({"K": K, "N": N, "M": M}){% endhighlight %}

<p>Bạn có thể rằng hiện tượng tương quan bị ẩn bằng việc thay <code>dcc</code> thành <code>d_sim</code> với các mô hình <code>m5_5</code>, <code>m5_6</code>, <code>m5_7</code>. Nhìn với tóm tắt của <code>print_summary</code> và bạn sẽ thấy cùng một hiện tượng ẩn đi khi slope trở nên mạnh hơn trong <code>m5_7</code>. Hai DAG khác cũng có thể được mô phỏng:</p>
<b>code 5.43</b>
{% highlight python %}# M -> K <- N
# N -> M
n = 100
N = dist.Normal().sample(random.PRNGKey(0), (n,))
M = dist.Normal(N).sample(random.PRNGKey(1))
K = dist.Normal(N - M).sample(random.PRNGKey(2))
d_sim2 = pd.DataFrame({"K": K, "N": N, "M": M})
# M -> K <- N
# M <- U -> N
n = 100
N = dist.Normal().sample(random.PRNGKey(3), (n,))
M = dist.Normal(M).sample(random.PRNGKey(4))
K = dist.Normal(N - M).sample(random.PRNGKey(5))
d_sim3 = pd.DataFrame({"K": K, "N": N, "M": M}){% endhighlight %}
<p>Trong ví dụ sữa của các loài khỉ, có thể tương quan dương giữa thể trọng lớn và tỉ lệ vỏ não xuất phát từ việc đánh đổi giữa tuổi thọ và khả năng học hỏi. Sinh vật lớn thường sống lâu hơn. Và trong sinh vật này, sự đầu tư về học hỏi có thể là một đầu tư tốt hơn, bởi vì học hỏi có thể được tận dụng tốt trên tuổi thọ dài hơn. Cả thể trọng lớn và tỉ lệ vỏ não lớn sau đó ảnh hưởng đến thành phần sữa, nhưng theo hướng khác nhau, với lý do khác nhau. Câu chuyện này suy ra rằng DAG có mũi tên $M \to N$, DAG đầu tiên, là đúng. Nhưng với những bằng chứng trong tay, chúng ta không dễ để thấy cái nào là đúng. Để tìm ra các DAG <strong>TƯƠNG ĐỒNG MARKOV</strong>, đầu tiên ta định nghĩa DAG và nhờ code làm chuyện nặng nhọc đó:</p>
<b>code 5.44</b>
{% highlight python %}from itertools import combinations
from causalgraphicalmodels import CausalGraphicalModel as CGM
def markov_equivalence(input_dag):
    '''
    :param input_dag:
        a CausalGraphicalModel object, from package causalgraphicalmodels.
    :return list:
        list of CausalGraphicalModel objects with same independence relationships, or a set of MARKOV EQUIVALENCE.
    '''
    independence = input_dag.get_all_independence_relationships()
    edges = list(input_dag.dag.edges)
    nodes = list(input_dag.dag.nodes)
    MElist = []
    for i in range(len(edges)+1):
        for j in combinations(range(len(edges)), i):
            edges2 = edges.copy()
            for elem in j:
                edges2[elem] = np.flip(edges2[elem])
            try:
                dag = CGM(nodes=nodes, edges=edges2)
                if dag.get_all_independence_relationships() == independence:
                    MElist += [dag]
            except AssertionError:
                pass # not a dag assertion
    print('Markov Equivalence set of origin dag:')
    print('\tNodes:', end=" "); print(*nodes, sep=", ")
    print('\tEdges:', end=" "); print(*edges, sep=", ")
    print('\tConditional Independencies:', independence, end='\n\n')
    for idx, cgm in enumerate(MElist):
        print(f"CGM {idx} edges:\n{cgm.dag.edges}\n")
    return MElist
{% endhighlight %}
<p>Kết quả của hàm <code>markov_equivalence</code> của DAG đầu tiên sẽ gồm 6 DAG khác nhau. Bạn nên loại trừ DAG nào, dựa vào kiến thức khoa học của các biến số.</p></div>

## <center>5.3 Biến phân nhóm</center><a name="a3"></a>

Một câu hỏi thường gặp trong các phương pháp thống kê là biến kết cục  thay đổi như thế nào khi có hoặc không có sự hiện biến của một phân nhóm. Một phân nhóm tức là rời rạc và không có thứ tự. Ví dụ, lần nữa xem xét các loài khác nhau trong data năng lượng sữa. Vài loài trong chúng là vượn, vài loài là khỉ Tân Thế Giới. Chúng ta muốn hỏi rằng dự đoán sẽ thay đổi như thế nào khi sinh vật là loài vượn thay vì loài khỉ. Nhóm loài là **BIẾN PHÂN NHÓN (CATEGORICAL VARIABLE)**, bởi vì không thể có loài nửa vượn nửa khỉ (tính rời rạc), và không có loài nào lớn hơn hoặc nhỏ hơn loài khác (tính thứ tự). Một số ví dụ khác của biến phân nhóm:

- Giới tính: nam, nữ
- Trạng thái phát triển: sơ sinh, thiếu niên, trưởng thành 
- Địa lý: Châu Phi, Châu Âu,..

Nhiều bạn đọc có thể đã biết loại biến này, thường được gọi là **YẾU TỐ (FACTOR)**, có thể cho vào mô hình tuyến tính dễ dàng. Nhưng nó không được biết nhiều về cách những biến này được đại diện như thế nào trong mô hình. Máy tính làm tất cả công việc này cho chúng ta, giấu đi cỗ máy đằng sau. Nhưng vẫn có vài điểm hay khiến nó đáng để phơi bày cỗ máy ra. Biết được cỗ máy (golem) làm việc như thế nào sẽ giúp bạn diễn giải phân phối posterior và cho bạn nhiều sức mạnh để xây dụng mô hình.

<div class="alert alert-info">
<p><strong>Các quốc gia liên tục.</strong> Với phần mềm tự động hoá nhưng thiếu sự quan tâm đúng mực, biến phân nhóm có thể là nguy hiểm. Năm 2015, một nghiên cứu trên tạp chí có tiếng, trên 1170 trẻ từ sáu quốc gia, cho thấy có mối tương quan âm mạnh giữa tôn giáo và lòng tốt bụng.<sup><a name="r85" href="#85">85</a></sup> Bài báo đã gây ra một làn sóng trong giới khoa học tôn giáo, bởi vì nó ngược lại với văn học hiện thời. Sau khi tái phân tích, lỗi được tìm ra là biến quốc gia, là biến phân nhóm, được nhập vào như là biến liên tục. Nó làm cho Canada (giá trị 2) gấp hai lần "quốc gia" hơn so với nước Mỹ (giá trị 1). Sau khi tái phân tích với quốc gia là biến phân nhóm, kết quả trái ngược này mất đi và bài báo bị rút lại. Đây là một kết thúc tốt đẹp, bởi vì tác giả chia sẻ data của họ. Bao nhiêu trường hợp giống vậy đã xảy ra, chưa được phát hiện bởi data không bao giờ được chia sẻ và có khả năng bị mất vĩnh viễn?</p></div>

### 5.3.1 Nhóm nhị phân

Trong trường hợp đơn giản nhất, biến quan tâm chỉ có 2 nhóm, như *nam* và *nữ*. Hãy quay lại bộ data Kalahari bạn đã gặp ở Chương 4. Lúc đó, chúng ta đã bỏ mặc giới tính khi dự đoán chiều cao, mặc dù rõ ràng chúng ta mong đợi nam và nữa có chiều cao trung bình khác nhau. Hãy nhìn vào những biến có sẵn:

<b>code 5.45</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv?raw=true", sep=";")
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
      <th>0</th>
      <td>151.765</td>
      <td>47.825606</td>
      <td>63.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139.700</td>
      <td>36.485807</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>136.525</td>
      <td>31.864838</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>156.845</td>
      <td>53.041915</td>
      <td>41.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145.415</td>
      <td>41.276872</td>
      <td>51.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table></samp></p>

Biến `male` đây là một biến dự đoán mới, còn được gọi là **BIẾN CHỈ ĐIỂM (INDICATOR VARIABLE)**. Biến chỉ điểm - đôi khi được gọi là biến "dummy" - là thiết bị để mã hoá phân nhóm không có thứ tự thành những mô hình định lượng. Ở đây không có khái niệm *nam* nhiều hơn *nữ* 1 giá trị. Mục đích của biến `male` là chỉ điểm người đó trong mẫu có phải *nam* hay không. Nên nó có giá trị 1 khi người đó là *nam*, còn nhóm khác là giá trị 0. Không quan trọng nhóm nào được chỉ điểm là 1. Mô hình không quan tâm điều đó. Nhưng để diễn giải mô hình chính xác cần bạn phải nhớ nhóm đó, cho nên thông thường người ta đặt tên biến theo phân nhóm được gán giá trị 1.

Có 2 cách để tạo mô hình từ thông tin này. Đầu tiên là dùng trực tiếp *indicator variable* trong mô hình tuyến tính, xem như nó là một biến dự đoán thông thường. Hiệu ứng của biến chỉ điểm là mở một tham số vào những quan sát trong nhóm đó. Đồng thời, biến đó sẽ tắt tham số cho những quan sát trong nhóm khác. Nó sẽ rõ hơn nếu bạn nhìn vào định nghĩa toán học của mô hình. Xem xét lại mô hình tuyến tính của chiều cao, như trong Chương 4. Bây giờ chúng ta sẽ không quan tâm cân nặng và biến khác và chỉ tập trung vào giới tính.

$$\begin{aligned}
h_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha + \beta_m m_i \\
\alpha &\sim \text{Normal}(178, 20) \\
\beta_m &\sim \text{Normal}(0, 10) \\
\sigma &\sim \text{Uniform}(0, 50)\\
\end{aligned}$$

Trong đó $h$ là chiều cao và $m$ là biến dummy chỉ điểm cho một cá thể nam. Tham số $\beta_m$ chỉ ảnh hưởng dự đoán cho những trường hợp khi $m_i=1$. Khi $m_i=0$, nó không có hiệu ứng lên dự đoán, bởi vì nó nhân với zero trong mô hình tuyến tính, $\alpha +\beta_m m_i$, bị mất đi, cho dù là giá trị nào. Ở đây chỉ muốn nói rằng, khi $m_i=1$, mô hình tuyến tính là $\mu_i=\alpha + \beta_m$. Và khi $m_i=0$, mô hình tuyến tính đơn giản là $\mu_i=\alpha$.

Sử dụng cách tiếp cận này nghĩa là $\beta_m$ đại diện cho trung bình *mong đợi* giữa chiều cao nam và nữ. Tham số $\alpha$ dùng để dự đoán cho cả chiều cao nam và nữa. Nhưng chiều cao nam có thêm $\beta_m$. Điều này nghĩa là $\alpha$ không còn là chiều cao trung bình trong mẫu, mà là chiều cao trung bình của nữ. Nó có thể làm cho việc gán các prior hợp lý khó khăn hơn. Nếu bạn không có biết hiệu số mong đợi về chiều cao - cái gì sẽ hợp lý trước khi thấy data? - thì cách tiếp cận này có thể là một phiền phức. Dĩ nhiên bạn có thể tránh nó ra bằng prior mơ hồ - vì có quá nhiều data.

Một hệ quả của việc gán prior cho hiệu số là các tiếp cận này đã giả định rằng có nhiều tính bất định hơn về một nhóm - "nam" trong trường hợp này - so với nhóm khác. Tại sao? Bởi vì dự đoán cho nam bao gồm hai tham số và do đó có hai prior. Chúng ta có thể mô phỏng điều này trực tiếp từ prior. Phân phối prior cho $\mu$ cho nam và nữa là:

<b>code 5.46</b>
```python
mu_female = dist.Normal(178, 20).sample(random.PRNGKey(0), (int(1e4),))
diff = dist.Normal(0, 10).sample(random.PRNGKey(1), (int(1e4),))
mu_male = dist.Normal(178, 20).sample(random.PRNGKey(2), (int(1e4),)) + diff
print_summary({"mu_female": mu_female, "mu_male": mu_male}, 0.89, False)
```
<samp>             mean    std  median    5.5%   94.5%     n_eff  r_hat
mu_female  178.21  20.22  178.24  147.19  211.84   9943.61   1.00
  mu_male  178.10  22.36  178.51  142.35  213.41  10190.57   1.00</samp>

Prior cho *nam* thì rộng hơn, bởi vì nó dùng cả hai tham số. Mặc dù trong hồi quy đơn giản này, các prior sẽ trôi đi nhanh chóng, nhưng nói chung chúng ta nên cẩn thận. Chúng ta thực ra không phải không chắc chắn về chiều cao nam hơn chiều cao nữ, *kiến thức tiền nghiệm (a priori)*. Có cách nào khác không?

Một cách khác có sẵn cho chúng ta là dùng **BIẾN CHỈ SỐ (INDEX VARIABLE)**. Biến chỉ số gồm các số nguyên tương ứng cho các nhóm khác nhau. Số nguyên chỉ là tên, nhưng nó cho phép chúng ta tham khảo đến danh sách các tham số tương ứng, mỗi một cho từng nhóm. Trong trường hợp này, chúng ta có thể tạo ra chỉ số như sau:

<b>code 5.47</b>
```python
d["sex"] = jnp.where(d.male.values == 1, 1, 0)
d.sex
```
<samp>0      1
1      0
2      0
3      1
4      0
      ..
539    1
540    1
541    0
542    1
543    1
Name: sex, Length: 544, dtype: int32</samp>

Bây giờ "0" nghĩa là nữ, "1" nghĩa là nam. Không có thứ tự được đặt ra. Chúng chỉ là các nhãn. Và dạng toán học của mô hình trở thành:

$$\begin{aligned}
h_i  &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{\text{sex}[i]} \\
\alpha_j &\sim \text{Normal}(178, 20) && \text{for} \; j = 0..1 \\
\sigma &\sim \text{Uniform}(0, 50)\\
\end{aligned}$$

Chúng ta đã tạo ra một danh sách gồm các tham số $\alpha$, mỗi một cho mỗi giá trị độc nhất trong biến chỉ số. Cho nên trong trường hợp này chúng ta có được hai tham số $\alpha$, tên là $\alpha_0$ và $\alpha_1$. Các con số tương ứng với giá trị trong biến chỉ số `sex`. Tôi biết rằng điều này phức tạp, nhưng nó giải quyết vấn đề về prior của chúng ta. Bây giờ cùng một prior được gán cho mỗi một, tương ứng với quan điểm rằng tất cả các nhóm là như nhau, trước khi thấy data. Không nhóm nào có nhiều tính bất định tiền nghiệm hơn nhóm khác. Và bạn sẽ thấy chút nữa, các tiếp cận này mở rộng dễ dàng cho tình huống có nhiều hơn hai nhóm.

Hãy ước lượng posterior của mô hình trên, sử dụng biến chỉ số.

<b>code 5.48</b>
```python
def model(sex, height):
    a = numpyro.sample("a", dist.Normal(178, 20).expand([len(set(sex))]))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = a[sex]
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)


m5_8 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_8, optim.Adam(1), Trace_ELBO(), sex=d.sex.values, height=d.height.values)
p5_8, losses = svi.run(random.PRNGKey(0), 2000)
post = m5_8.sample_posterior(random.PRNGKey(1), p5_8, (1000,))
print_summary(post, 0.89, False)
```
<samp>           mean       std    median      5.5%     94.5%     n_eff     r_hat
 a[0]    135.02      1.63    135.07    132.32    137.46    931.50      1.00
 a[1]    142.56      1.73    142.54    140.02    145.51   1111.51      1.00
sigma     27.32      0.84     27.32     26.03     28.71    951.62      1.00</samp>

Chú ý rằng trong kết quả `print_summary` này sẽ cho toàn bộ tham số dạng vector hoặc ma trận. Đôi khi chúng ta rất nhiều tham số loại này và bạn không muốn nhìn thấy tất cả giá trị đó. Bạn sẽ hiểu tôi muốn nói gì ở các chương sau.

Diễn giải những tham số này là dễ - chúng là chiều cao mong đợi trong mỗi nhóm. Nhưng đôi khi chúng ta chỉ quan tâm đến hiệu giữa các nhóm. Trong trường hợp này, hiệu mong đợi giữa nam và nữ là gì? Chúng ta ó thể tính nó bằng cách dùng các mẫu trong posterior. Thực vậy, tôi sẽ trích xuất những mẫu posterior thành một DataFrame và thêm phép tính đó trực tiếp vào cùng một DataFrame:

<b>code 5.49</b>
```python
post = m5_8.sample_posterior(random.PRNGKey(1), p5_8, (1000,))
post["diff_fm"] = post["a"][:, 0] - post["a"][:, 1]
print_summary(post, 0.89, False)
```
<samp>            mean       std    median      5.5%     94.5%     n_eff     r_hat
   a[0]   135.02      1.63    135.07    132.32    137.46    931.50      1.00
   a[1]   142.56      1.73    142.54    140.02    145.51   1111.51      1.00
diff_fm    -7.54      2.38     -7.47    -11.77     -4.32    876.56      1.00
  sigma    27.32      0.84     27.32     26.03     28.71    951.62      1.00</samp>

Kết quả phép tính của chúng ta ở hàng cuối cùng, dưới dạng một tham số mới trong posterior. Nó là hiệu số mong đợi giữa chiều cao nam và nữ. Dạng phép tính này còn gọi là **TƯƠNG PHẢN (CONTRAST)**. Cho dù số lượng nhóm nhiều cỡ nào, bạn có thể dùng mẫu từ posterior để tính tương phản giữa hai nhóm bất kỳ.

### 5.3.2 Nhiều nhóm

Nhóm nhị phân thì dễ dàng, cho dù bạn dùng biến chỉ điểm hay biến chỉ số. Nhưng khi có nhiều hơn hai nhóm, biến chỉ điểm sẽ bùng nổ. Bạn phải cần một biến chỉ điểm cho mỗi nhóm. Nếu bạn có $k$ nhóm độc nhất, bạn cần đến $k-1$ biến chỉ điểm. Nhiều công cụ tự động thực ra dùng cách tiếp cận này, tạo $k-1$ biến chỉ điểm cho bạn và trả về $k-1$ tham số (kèm theo với intercept).

Nhưng chúng ta sẽ tiếp cận bằng biến chỉ số. Nó vẫn không thay đổi khi bạn thêm nhiều nhóm hơn. Chắc chắn là bạn có nhiều tham số hơn, chỉ có có nhiều như cách tiếp cận dùng biến chỉ điểm. Nhưng thông số mô hình cũng nhìn giống như trong trường hợp nhị phân. Và prior cũng dễ dàng, trừ phi bạn thực sự có thông tin prior về tương phản. Việc quen dần với cách tiếp cận bằng biến chỉ số rất quan trọng, bởi vì nhiều mô hình đa tầng (Chương 13) phụ thuộc vào nó.

Hãy khám phá ví dụ sữa các loài khỉ lần nữa. Chúng ta bây giờ quan tâm đến biến `clade`, tức là mã hoá cho tên của các loài:

<b>code 5.50</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv?raw=true", sep=";")
d.clade.unique()
```
<samp>['Strepsirrhine', 'New World Monkey', 'Old World Monkey', 'Ape']</samp>

Chúng ta muốn có chỉ số cho mỗi một trong bốn nhóm này. Bạn có thể làm điều này bằng tay, nhưng việc gán con số vào chúng sẽ dễ hơn:

<b>code 5.51</b>
```python
d["clade_id"] = d.clade.astype("category").cat.codes
```

Hãy sử dụng mô hình để đo lường năng lượng trung bình của sữa tại mỗi nhóm. Ở dạng toán học:

$$\begin{aligned}
K_i &\sim \text{Normal}(\mu_i, \sigma) \\
\mu_i &= \alpha_{clade[i]}\\
\alpha_j &\sim \text{Normal}(0, 0.5) && \text{for} \; j = 0..3\\
\sigma &\sim \text{Exponential}(1) \\
\end{aligned} $$

Nhớ lại rằng, $K$ là kilocalory được chuẩn hoá. Tôi sẽ mở rộng thêm một chút prior của $\alpha$, để cho phép sự khác nhau giữa các nhóm được đa dạng hoá, nếu data cho phép chúng. Nhưng tôi khuyến khích bạn thay đổi prior đó và tái ước lượng posterior nhiều lần để bạn có thể thấy sự khác nhau giữa posterior giữa các nhóm phụ thuộc vào nó. Chạy `SVI`:

<b>code 5.52</b>
```python
d["K"] = d["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
def model(clade_id, K):
    a = numpyro.sample("a", dist.Normal(0, 0.5).expand([len(set(clade_id))]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a[clade_id]
    numpyro.sample("height", dist.Normal(mu, sigma), obs=K)
m5_9 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_9, optim.Adam(1), Trace_ELBO(), clade_id=d.clade_id.values, K=d.K.values)
p5_9, losses = svi.run(random.PRNGKey(0), 1000)
post = m5_9.sample_posterior(random.PRNGKey(1), p5_9, (1000,))
labels = ["a[" + str(i) + "]:" + s for i, s in enumerate(sorted(d.clade.unique()))]
az.plot_forest({"a": post["a"][None, ...]}, hdi_prob=0.89)
plt.gca().set(yticklabels=labels[::-1], xlabel="kcal mong đợi (chuẩn hoá)")
```

![](/assets/images/forest 5-3.svg)

Tôi sử dụng đối số `yticklabels` (không bắt buộc) để thay đổi tên của tham số `a[0]` đến `a[3]` với tên của các loài từ biến gốc. Trong thực hành, chúng ta nên cẩn thận khi theo dõi những giá trị chỉ số nào cho nhóm nào. Đừng bao giờ tin tưởng vào biến chỉ số từ code là luôn luôn làm đúng.

Nếu bạn có một biến phân nhóm khác mà bạn muốn thêm vào mô hình, thì cách tiếp cận là như nhau. Ví dụ, hãy gán ngẫu nhiên những loài khỉ này vào bốn nhóm:[1] Gryffindor, [2] Hufflepuff, [3] Ravenclaw, và [4] Slytherin.

<b>code 5.53</b>
```python
key = random.PRNGKey(63)
d["house"] = random.choice(key, jnp.repeat(jnp.arange(4), 8), d.shape[:1], False)
```

Bây giờ chúng ta có thể thêm những nhóm này vào thành một biến dự đoán trong mô hình:

<b>code 5.54</b>
```python
def model(clade_id, house, K):
    a = numpyro.sample("a", dist.Normal(0, 0.5).expand([len(set(clade_id))]))
    h = numpyro.sample("h", dist.Normal(0, 0.5).expand([len(set(house))]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a[clade_id] + h[house]
    numpyro.sample("height", dist.Normal(mu, sigma), obs=K)
m5_10 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m5_10,
    optim.Adam(1),
    Trace_ELBO(),
    clade_id=d.clade_id.values,
    house=d.house.values,
    K=d.K.values,
)
p5_10, losses = svi.run(random.PRNGKey(0), 1000)
```

Nếu bạn nhìn vào posterior, bạn sẽ thấy Slytherin nổi bật hơn cả.

<div class="alert alert-info">
<p><strong>Hiệu số và ý nghĩa thống kê.</strong> Một lỗi thường gặp trong diễn giải ước lượng của tham số là cho rằng bởi vì một tham số đủ xa zero - tức là "có ý nghĩa (significant)"" - và một tham số khác thì không - "không có ý nghĩa (not significant)" - thì hiệu số giữa hai tham số này là có ý nghĩa. Điều này không phải luôn luôn đúng.<sup><a name="r86" href="#86">86</a></sup> Nó không phải là vấn đề riêng cho phân tích non-Bayes: Nếu bạn muốn biết được phân phối của hiệu số, thì bạn phải tự tính nó, <strong>SỰ TƯƠNG PHẢN (CONTRAST)</strong>. Chỉ quan sát thôi là không đủ, ví dụ, slope của nam bị chồng lắp rất nhiều với zero trong khi slope của nữ thì phần lớn trên zero. Bạn phải tính phân phối hiệu số của hai slope giữa nam và nữ. Ví dụ, giả sử bạn có phân phối posterior cho hai tham số, $\beta_f$ và $\beta_m$. Trung bình và độ lệch chuẩn của $\beta_f$ là $0.15 \pm 0.02$ và của $\beta_m$ là $0.02 \pm 0.10$. Cho nên trong khi $\beta_f$ là xa zero một cách đáng tin cậy (có ý nghĩa) và $\beta_m$ thì không, hiệu số giữa chúng (giả sử chúng không tương quan) là $(0.15-0.02) \pm \sqrt{0.02^2 + 0.1^2 } \approx 0.13 \pm 0.10$. Phân phối hiệu số này chồng lắp rất nhiều với zero. Nói cách khác, bạn có thể tự tin rằng $\beta_f$ xa zero, nhưng không thể khẳng định rằng hiệu số giữa $\beta_m$ và $\beta_f$ là xa zero.</p>
<p>Trong bối cảnh phép kiểm non-Bayes, hiện tượng này xuất hiện từ sự thật rằng ý nghĩa thống kê được suy luận rất mạnh từ một hướng: sự khác biệt so với giả thuyết vô hiệu. khi $\beta_m$ chồng lắp với zero, nó cũng có thể chồng lắp với giá trị rất xa zero. Giá trị của nó là bất định. Cho nên khi bạn so sánh $\beta_m$ và $\beta_f$, sự so sánh cũng bất định, thể hiện thông qua độ rộng của phân phối posterior hiệu số $\beta_f - \beta_m$. Ngầm bên dưới ví dụ này là một lỗi cơ bản trong diễn giải ý nghĩa thống kê: Lỗi chấp nhận giả thuyết vô hiệu. Mỗi khi một đề tài hay sách vở nói gì đó như "chúng ta không tìm thấy sự khác nhau" hay "không có hiệu ứng", điều này thường nghĩa là vài tham số không khác biệt có ý nghĩa với zero, và tác giả sử dụng zero như là điểm ước lượng. Điều này là phi logic và cực kỳ phổ biến.</p></div>

## <center>5.4 Tổng kết</center><a name="a4"></a>

Chương này giới thiệu hồi quy đa biến, một cách để xây dựng mô hình mô tả cho trung bình của một đại lượng đo lường liên quan như thế nào với một hay nhiều biến dự đoán. Câu hỏi định nghĩa của mô hình đa biến là: *Giá trị của việc biết mỗi biến dự đoán, khi chúng ta đã biết những biến dự đoán khác?* Đáp án cho câu hỏi này không tự nó cung cấp thông tin nhân quả. Suy luận nhân quả cần thêm nhiều giả định. Đồ thị có hướng không tuần hoàn (DAG) đơn giản dành cho mô hình nhân quả là một cách để đại diện cho những giả định đó. Trong chương sau chúng ta sẽ tiếp tục xây dựng khung quy trình DAG và xem khi thêm biến dự đoán mới có thể tạo ra nhiều rắc rối giống như số vấn đề mà nó có thể giải quyết được.

---

<details><summary>Endnotes</summary>
<ol class="endnotes">
<li><a name="79" href="#r79">79. </a>“How to Measure a Storm’s Fury One Breakfast at a Time.” The Wall Street Journal: September 1, 2011.</li>
<li><a name="80" href="#r80">80. </a>See Meehl (1990), in particular the “crud factor” described on page 204.</li>
<li><a name="81" href="#r81">81. </a>Debates about causal inference go back a long time. David Hume is key citation. One curious obstacle in modern statistics is that classic causal reasoning requires that if A causes B, then B will always appear when A appears. But with probabilistic relationships, like those described in most contemporary scientific models, it is unsurprising to talk about probabilistic causes, in which B only sometimes follows A. See http://plato.stanford.edu/entries/causation-probabilistic/.</li>
<li><a name="82" href="#r82">82. </a>See Pearl (2014) for an accessible introduction, with discussion. See also Rubin (2005) for a related approach. An important perspective missing in these is an emphasis on rigorous scientific models that make precise predictions. This tension builds throughout the book and asserts itself in Chapter 16.</li>
<li><a name="83" href="#r83">83. </a>See Freckleton (2002).</li>
<li><a name="84" href="#r84">84. </a>Data from Table 2 of Hinde and Milligan (2011).</li>
<li><a name="85" href="#r85">85. </a>See Decety et al. (2015) for the original and retraction notice. See Shariff et al. (2016) for the reanalysis.</li>
<li><a name="86" href="#r86">86. </a>See Gelman and Stern (2006) for further explanation, and see Nieuwenhuis et al. (2011) for some evidence of how commonly this mistake occurs.</li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
<p><strong>5E1.</strong> Which of the linear models below are multiple linear regressions?</p>
<ol>
<li>$\mu_i = \alpha + \beta x_i$</li>
<li>$\mu_i = \beta_x x_i + \beta_z z_i$</li>
<li>$\mu_i = \alpha + \beta(x_i − z_i)$</li>
<li>$\mu_i = \alpha + \beta_x x_i + \beta_z z_i$</li>
</ol>
<p><strong>5E2.</strong> Write down a multiple regression to evaluate the claim: <i>Animal diversity is linearly related to latitude, but only after controlling for plant diversity.</i> You just need to write down the model definition.</p>
<p><strong>5E3.</strong> Write down a multiple regression to evaluate the claim: <i>Neither amount of funding nor size of laboratory is by itself a good predictor of time to PhD degree; but together these variables are both positively associated with time to degree.</i> Write down the model definition and indicate which side of zero each slope parameter should be on.</p>
<p><strong>5E4.</strong> Suppose you have a single categorical predictor with 4 levels (unique values), labeled A, B, C and D. Let $A_i$ be an indicator variable that is 1 where case i is in category A. Also suppose $B_i$, $C_i$, and $D_i$ for the other categories. Now which of the following linear models are inferentially equivalent ways to include the categorical variable in a regression? Models are inferentially equivalent when it’s possible to compute one posterior distribution from the posterior distribution of another model.</p>
<ol>
<li>$\mu_i = \alpha + \beta_A A_i + \beta_B B_i + \beta_D D_i$</li>
<li>$\mu_i = \alpha + \beta_A A_i + \beta_B B_i + \beta_C C-i + \beta_D D_i$</li> 
<li>$\mu_i = \alpha + \beta_B B_i + \beta_C C_i + \beta_D D_i$</li>
<li>$\mu_i = \alpha_A A_i + \alpha_B B_i + \alpha_C C_i + \alpha_D D_i$</li>
<li>$\mu_i = \alpha_A (1 − B_i − C_i − D_i) + \alpha_B B_i + \alpha_C C_i + \alpha_D D_i$</li></ol>
<p><strong>5M1.</strong> Invent your own example of a spurious correlation. An outcome variable should be correlated with both predictor variables. But when both predictors are entered in the same model, the correlation between the outcome and one of the predictors should mostly vanish (or at least be greatly reduced).</p>
<p><strong>5M2.</strong> Invent your own example of a masked relationship. An outcome variable should be correlated with both predictor variables, but in opposite directions. And the two predictor variables should be correlated with one another.</p>
<p><strong>5M3.</strong> It is sometimes observed that the best predictor of fire risk is the presence of firefightersStates and localities with many firefighters also have more fires. Presumably firefighters do not cause fires. Nevertheless, this is not a spurious correlation. Instead fires cause firefighters. Consider the same reversal of causal inference in the context of the divorce and marriage data. How might a high divorce rate cause a higher marriage rate? Can you think of a way to evaluate this relationship, using multiple regression?</p>
<p><strong>5M4.</strong> In the divorce data, States with high numbers of members of the Church of Jesus Christ of Latter-day Saints (LDS) have much lower divorce rates than the regression models expected. Find a list of LDS population by State and use those numbers as a predictor variable, predicting divorce rate using marriage rate, median age at marriage, and percent LDS population (possibly standardized). You may want to consider transformations of the raw percent LDS variable.</p>
<p><strong>5M5.</strong> One way to reason through multiple causation hypotheses is to imagine detailed mechanisms through which predictor variables may influence outcomes. For example, it is sometimes argued that the price of gasoline (predictor variable) is positively associated with lower obesity rates (outcome variable). However, there are at least two important mechanisms by which the price of gas could reduce obesity. First, it could lead to less driving and therefore more exercise. Second, it could lead to less driving, which leads to less eating out, which leads to less consumption of huge restaurant meals. Can you outline one or more multiple regressions that address these two mechanisms? Assume you can have any predictor data you need.</p>
<p><strong>5H1.</strong> In the divorce example, suppose the DAG is: $M \to A \to D$. What are the implied conditional independencies of the graph? Are the data consistent with it?</p>
<p><strong>5H2.</strong> Assuming that the DAG for the divorce example is indeed $M \to A \to D$, fit a new model and use it to estimate the counterfactual effect of halving a State’s marriage rate M. Use the counterfactual example from the chapter as a template.</p>
<p><strong>5H3.</strong> Return to the milk energy model, <code>m5_7</code>. Suppose that the true causal relationship among the variables is:</p>
<img src="./assets/images/dag 5-6.svg">
<p>Now compute the counterfactual effect on K of doubling M. You will need to account for both the direct and indirect paths of causation. Use the counterfactual example from the chapter as a template.</p>
<p><strong>5H4.</strong> Here is an open practice problem to engage your imagination. In the divorce date, States in the southern United States have many of the highest divorce rates. Add the South indicator variable to the analysis. First, draw one or more DAGs that represent your ideas for how Southern American culture might influence any of the other three variables ($D$, $M$ or $A$). Then list the testable implications of your DAGs, if there are any, and fit one or more models to evaluate the implications. What do you think the influence of “Southerness” is?</p>
</details>


