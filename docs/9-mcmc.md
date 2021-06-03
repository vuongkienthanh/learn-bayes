---
title: "Chapter 9: Markov Chain Monte Carlo"
description: "Chương 9: Markov Chain Monte Carlo"
---

- [9.1 Vương quốc đảo của Đức Vua Markov](#a1)
- [9.2 Thuật toán Metropolis](#a2)
- [9.3 Hamiltonian Monte Carlo](#a3)
- [9.4 HMC đơn giản](#a4)
- [9.5 Chăm sóc và nuôi dưỡng chuỗi Markov](#a5)
- [9.6 Tổng kết](#a6)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import inspect
import math
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import lax, ops, random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation{% endhighlight %}</details>

Vào thế kỷ 20, các nhà khoa học và kỹ sư bắt đầu xuất bản sách về những con số ngẫu nhiên ([**HÌNH 9.1**](#f1)). Với những nhà khoa học từ thế kỷ trước, những cuốn sách này trông điên rồ. Với hầu hết lịch sử phương Tây, xác suất từng là kẻ thù. Điển hình như Rome, xác suất được hình tượng hoá bởi Fortuna, nữ thần của định mệnh, với vòng xoay của may rủi. Trái ngược với Fortuna là Minerva, nữ thần trí tuệ và tinh thông. Chỉ những người tuyệt vọng mới cầu nguyện Fortuna, trong khi mọi người cầu khẩn Minervà giúp đỡ. Chắc chắn khoa học là lĩnh vực của Minerva, một lãnh địa mà không có vai trò hữu ích của Fortuna.

Nhưng ở thế kỷ 20, Fortuna và Minerva đã thành cộng sự của nhau. Bây giờ một vài trong chúng ta chắc sẽ hoang mang bởi ý kiến rằng hiểu biết xác suất có thể giúp chúng ta đạt được tri trức. Mọi thứ từ dự báo thời tiết đến kinh tế đến sinh học tiến hoá là ưu thế bởi khoa học của tiến trình ngẫu nhiên. Nhà khoa học dựa vào những con số ngẫu nhiên để thiết kế thí nghiệm thích hợp. Và nhà toán học thường xuyên sử dụng nhập liệu ngẫu nhiên để tính toán kết quả cụ thể.

Chương này giới thiệu một ví dụ thường gặp của cặp đôi Fortuna và Minerva: ước lượng phân phối xác suất posterior bằng tiến trình phân phối ngẫu nhiên là **MARKOV CHAIN MONTE CARLO (MCMC)**. Không giống như những chương trước trong sách này, ở đây chúng ta sẽ tạo mẫu từ posterior kết hợp mà không cần tối đa hoá gì cả. Thay vì phải dựa vào kỹ thuật như quadratic để ước lượng hình dạng của posterior, bây giờ chúng ta sẽ có thể lấy mẫu trực tiếp từ posterior mà không cần giả định nó là Gaussian, hay hình dạng khác.

Cái giá của sức mạnh này là nó cần nhiều thời gian hơn để hoàn thành ước lượng, và thông thường nhiều việc cũng cần thiết phải có để xác định mô hình. Nhưng ích lợi là bạn có khả năng ước lượng mô hình trực tiếp, như những mô hình tuyến tính tổng quát và mô hình đa tầng ở các chương sau. Những mô hình này hầu như tạo ra phân phối posterior không phải Gaussian, và đôi khi chúng không có cách nào ước lượng được với những kỹ thuật ở các chương trước.

Một tin tốt là những công cụ xây dựng và kiểm tra ước lượng MCMC đang hoàn thiện theo thời gian. Trong sách này bạn sẽ gặp một phương pháp tiện lợi để chuyển công thức `quap` bạn đang sử dụng thành chuỗi Markov. Cỗ máy cho phép điều này khả thi là **STAN** (miễn phí và online tại: mc-stan.org). Tác giá của Stan mô tả nó là một "ngôn ngữ lập trình xác suất dùng để suy luận thống kê". Bạn sẽ không cần làm việc trực tiếp với Stan - gói `rethinking` cung cấp công cụ để sử dụng nó gián tiếp. Nhưng khi bạn học tiếp những kỹ thuật cao cấp hơn, bạn sẽ có thể tạo ra phiên bản Stan của mô hình mà bạn hiểu. Sau đó bạn có thể tuỳ biến chúng và chứng kiến sức mạnh của một Stan được trang bị đầy đủ.

<a name="f1"></a>![](/assets/images/fig 9-1.png)
<details class="fig"><summary>Hình 9.1: Một trang từ <i>A million Random Digits</i>, một cuốn sách không có gì ngoài số ngẫu nhiên.</summary></details>

<div class="alert alert-info">
<p><strong>Stan là một người.</strong> Ngôn ngữ lập trình Stang không phải là một từ viết tắt. Mà là, nó được đặt tên theo Stanislaw Ulam (1909-1984). Ulam được coi là một trong những người tạo ra Markov chain Monte Carlo. Cùng với Ed Teller, Ulam ứng dụng nó vào thiết kế bom nhiệt hạch. Nhưng ông ta và nhiều người khác sau đó ứng dụng phương pháp Monte Carlo tổng quát vào nhiều vấn đề đa dạng và ít quái dị hơn. Ulam đã có nhiều cống hiến quan trọng cho toán học thuần tuý, thuyết hỗn độn và sinh học phân tử.</p></div>

## <center>9.1 Vương quốc đảo của Đức Vua Markov</center><a name="a1"></a>

Bây giờ, hãy quên đi mật độ posterior và MCMC. Thay vào đó hãy xem ví dụ của Đức Vua Markov. Vua Markov là một nhà cai trị nhân từ của một vương quốc đảo, một quần đảo hình tròn, với 10 hòn đảo. Mỗi hòn đảo được lân cận bởi hai hòn đảo khác, và toàn bộ quần đảo tạo thành một vòng tròn. Các hòn đảo có kích thước khác nhau, và do đó có dân số với kích thước khác nhau sống trên đó. Đảo thứ hai đông dân gấp đôi hòn đảo thứ nhất, hòn đảo thứ ba đông dân gấp ba lần hòn đảo thứ nhất, v.v., cho đến hòn đảo lớn nhất, đông gấp 10 lần hòn đảo nhỏ nhất.

Đức Vua là một kẻ chuyên quyền, nhưng ông có một số nghĩa vụ đối với người dân của mình. Trong số các nghĩa vụ này, Vua Markov đồng ý thỉnh thoảng đến thăm từng hòn đảo trong vương quốc của mình. Vì người dân yêu mến vị vua của họ, nên mỗi hòn đảo muốn ông đến thăm họ thường xuyên hơn. Và vì vậy mọi người đều đồng ý rằng nhà vua nên đến thăm từng hòn đảo tương ứng với quy mô dân số của nó, ví dụ như đến thăm hòn đảo lớn nhất thường xuyên hơn gấp 10 lần so với hòn đảo nhỏ nhất.

Tuy nhiên, Đức Vua Markov không phải là người thích lập lịch trình hay giữ sổ sách, vì vậy ông ta muốn có một cách để hoàn thành nghĩa vụ của mình mà không cần lên kế hoạch cho các chuyến du lịch trước thời hạn nhiều tháng. Ngoài ra, vì quần đảo là một vành đai, nên nhà vua nhấn mạnh rằng ông chỉ di chuyển giữa các đảo liền kề, để giảm thiểu thời gian dành trên biển — giống như nhiều công dân trong vương quốc của mình, nhà vua tin rằng có những con quái vật biển ở giữa quần đảo.

Cố vấn của nhà vua, ông Metropolis, đã thiết kế một giải pháp thông minh cho những yêu cầu này. Chúng tôi sẽ gọi giải pháp này là thuật toán Metropolis. Đây là cách nó hoạt động.

1. Dù ở đâu, mỗi tuần, nhà vua quyết định giữa việc ở lại thêm một tuần nữa hay chuyển đến một trong hai hòn đảo liền kề. Để quyết định, anh ta tung một đồng xu.
2. Nếu đồng xu quay ngửa, nhà vua xem xét di chuyển đến đảo liền kề theo chiều kim đồng hồ quanh quần đảo. Nếu đồng xu lật ngửa, thay vào đó anh ta sẽ cân nhắc chuyển động ngược chiều kim đồng hồ. Gọi đảo mà đồng xu đề cử là đảo *đề xuất*.
3. Bây giờ, để xem liệu ông ta có di chuyển đến hòn đảo đề xuất hay không, Vua Markov đếm một lượng vỏ sò bằng với quy mô dân số tương đối của hòn đảo đề xuất. Ví dụ, nếu đảo đề xuất là số 9, thì anh ta sẽ đếm ra 9 vỏ sò. Sau đó, anh ta cũng đếm ra một số lượng đá bằng với dân số tương đối của hòn đảo hiện tại. Vì vậy, ví dụ, nếu hòn đảo hiện tại là số 10, thì Vua Markov sẽ giữ 10 viên đá, ngoài 9 vỏ sò.
4. Khi có nhiều vỏ sò hơn đá, Vua Markov luôn di chuyển đến hòn đảo đề xuất. Nhưng nếu có ít vỏ sò hơn đá, anh ta trừ số đá bằng số vỏ. Ví dụ, nếu có 4 vỏ và 6 viên đá, ông ta kết thúc với 4 vỏ và 6 - 4 = 2 viên đá. Sau đó, ông ta đặt những vỏ sò và những viên đá còn lại vào một chiếc túi. Anh ta đưa tay vào và lấy ra ngẫu nhiên một vật. Nếu nó là một vỏ sò, ông ta di chuyển đến hòn đảo đề xuất. Nếu không, ông ta sẽ ở lại thêm một tuần nữa. Kết quả là, xác suất ông ta di chuyển bằng số vỏ chia cho số viên đá ban đầu.

Quy trình này có vẻ kỳ dị và thành thật mà nói, hơi điên rồ. Nhưng nó hoạt động. Nhà vua sẽ di chuyển giữa các hòn đảo một cách ngẫu nhiên, đôi khi ở trên một hòn đảo trong nhiều tuần, những lần khác nhảy qua lại mà không có khuôn mẫu rõ ràng. Nhưng về lâu dài, quy trình này, bảo đảm rằng nhà vua sẽ được tìm thấy trên mỗi hòn đảo tỉ lệ với quy mô dân số của nó.

Bạn có thể tự chứng minh điều này, bằng cách mô phỏng cuộc hành trình của Vua Markov. Đây là một đoạn mã ngắn để thực hiện việc này, lưu trữ lịch sử cuộc hành trình của nhà vua ở vector `positions`:

<b>code 9.1</b>
```python
num_weeks = int(1e5)
positions = jnp.repeat(0, num_weeks)
current = 10
def body_fn(i, val):
    positions, current = val
    # record current position
    positions = ops.index_update(positions, i, current)
    # flip coin to generate proposal
    bern = dist.Bernoulli(0.5).sample(random.fold_in(random.PRNGKey(0), i))
    proposal = current + (bern * 2 - 1)
    # now make sure he loops around the archipelago
    proposal = jnp.where(proposal < 1, 10, proposal)
    proposal = jnp.where(proposal > 10, 1, proposal)
    # move?
    prob_move = proposal / current
    unif = dist.Uniform().sample(random.fold_in(random.PRNGKey(1), i))
    current = jnp.where(unif < prob_move, proposal, current)
    return positions, current
positions, current = lax.fori_loop(0, num_weeks, body_fn, (positions, current))
```

Tôi đã thêm comment vào đoạn code để giúp bạn giải mã nó. Ba dòng đầu tiên là định nghĩa số tuần để mô phỏng, một vector lịch sử trống, và vị trí hòn đảo khởi đầu (hòn đảo lớn nhất, số 10). Sau đó vòng lặp `lax.fori_loop` sẽ đi qua các tuần. Ở mỗi tuần, nó ghi nhận vị trí hiện tại của Đức Vua. Sau đó nó mô phỏng một lần tung đồng xu để đề cử một hòn đảo. Mánh ở đây nằm ở việc đảm bảo rằng nếu đề xuất ra "11", vòng lặp sẽ quay trở về hòn đảo 1 và đề xuất "0" sẽ thành hòn đảo 10. Cuối cùng, một số ngẫu nhiên giữa số không và số 1 được tạo ra từ `dist.Uniform`, và ông ta di chuyển, nếu số này nhỏ hơn tỉ số dân số của hòn đảo đề xuất và hòn đảo hiện tại (`proposal/current`).

Bạn có thể kết quả của mô phỏng này ở [**HÌNH 9.2**](#f2). Biểu đồ bên trái cho thấy vị trí của nhà vua trong 100 tuần mô phỏng du hành đầu tiên.

<b>code 9.2</b>
```python
plt.plot(range(1, 101), positions[:100], "o", mfc="none")
```

<a name="f2"></a>![](/assets/images/fig 9-2.svg)
<details class="fig"><summary>Hình 9.2: Kết quả của nhà vua khi đi theo thuật toán Metropolis. Biểu đồ bên trái cho thấy vị trí của nhà vua (trục tung) qua các tuần (trục hoành). Trong một tuần bất kỳ, nó gần như bất khả thi để nói rằng vị trí của nhà vua ở đâu. Biểu đồ bên phải cho thấy hiệu ứng lâu dài của thuật toán, khi thời gian trên mỗi hòn đảo lại tỉ lệ thuận với quy mô dân số của nó.</summary>
{% highlight python %}_, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].plot(range(1, 101), positions[:100], "o", mfc="none")
axes[0].set(xlabel="tuần", ylabel='đảo')
axes[1].hist(positions, bins=range(1, 12), rwidth=0.1, align="left")
axes[1].set(xlabel='đảo', ylabel='số tuần')
plt.tight_layout(){% endhighlight %}</details>

Khi bạn nhìn từ trái sang phải trong biểu đồ này, những điểm này cho thấy vị trí của nhà vua theo thời gian. Nhà vua di chuyển giữa các hòn đảo, hoặc đôi khi ở lại trong vài tuần. Biểu đồ này minh hoạ cho những con đường vô nghĩa của thuật toán Metropolis dành cho nhà vua. Biểu đồ bên trái lại cho thấy con đường này thực sự không vô nghĩa.

<b>code 9.3</b>
```python
plt.hist(positions, bins=range(1, 12), rwidth=0.1, align="left")
```

Trục hoành bây giờ là các hòn đảo (và dân số tương đối của chúng), trong khi trục dọc là số tuần nhà vua được tìm thấy trên mỗi hòn đảo. Sau toàn bộ 100.000 tuần (gần 2000 năm) của mô phỏng, bạn có thể thấy rằng tỷ lệ thời gian dành cho mỗi hòn đảo hội tụ gần như tỷ lệ chính xác với dân số tương đối của các hòn đảo.

Thuật toán sẽ vẫn hoạt động theo cách này, ngay cả khi chúng ta cho phép nhà vua có khả năng như nhau đề xuất chuyển đến bất kỳ hòn đảo nào từ bất kỳ hòn đảo nào, không chỉ giữa các hòn đảo lân cận. Miễn là Vua Markov vẫn sử dụng tỷ lệ dân số của hòn đảo đề xuất so với dân số của hòn đảo hiện tại làm xác suất di chuyển của mình, về lâu dài, ông sẽ dành khoảng thời gian thích hợp trên mỗi hòn đảo. Thuật toán cũng sẽ hoạt động đối với bất kỳ quần đảo lớn nào, ngay cả khi nhà vua không biết có bao nhiêu hòn đảo trong đó. Tất cả những gì anh ta cần biết tại bất kỳ thời điểm nào là dân số của hòn đảo hiện tại và dân số của hòn đảo đề xuất. Sau đó, không cần phải có bất kỳ kế hoạch định trước hoặc lưu giữ hồ sơ, Vua Markov vẫn có thể đáp ứng nghĩa vụ hoàng gia của mình để thăm dân của mình một cách tương xứng.

## <center>9.2 Thuật toán Metropolis</center><a name="a2"></a>

Thuật toán chính xác mà Vua Markov sử dụng là một trường hợp đặc biệt của thuật toán Metropolis tổng quát từ thế giới thực. Và thuật toán này là một ví dụ của Markov Chain Monte Carlo. Trong các ứng dụng thực tế, mục tiêu tất nhiên không phải là giúp một hoàng tộc tự động lập lịch trình cho hành trình của mình, mà thay vào đó là để lấy mẫu từ một phân phối đích thường không được xác định và phức tạp, như phân phối xác suất posterior.

- Các “đảo” trong mục tiêu của chúng ta là các giá trị tham số và chúng không cần phải rời rạc mà thay vào đó có thể nhận một loạt giá trị liên tục như bình thường.
- “Quy mô dân số” trong mục tiêu của chúng ta là xác suất posterior ở mỗi giá trị tham số.
- “Các tuần” trong mục tiêu của chúng tôi là các mẫu được lấy từ posterior kết hợp của các tham số trong mô hình.

Với điều kiện cách chúng ta chọn các giá trị tham số được đề xuất của mình ở mỗi bước là đối xứng — để có cơ hội như nhau trong việc đề xuất từ ​​A đến B và từ B đến A — thì thuật toán Metropolis cuối cùng sẽ cung cấp cho chúng ta một bộ tập hợp các mẫu từ posterior kết hợp. Sau đó, chúng ta có thể sử dụng các mẫu này giống như tất cả các mẫu bạn đã sử dụng trong cuốn sách này.

Thuật toán Metropolis là một ông lớn của nhiều chiến lược khác nhau để lấy mẫu từ các phân phối posterior chưa biết. Trong phần còn lại của phần này, tôi giải thích ngắn gọn khái niệm đằng sau phương pháp lấy mẫu Gibbs. Lấy mẫu Gibbs tốt hơn nhiều so với Metropolis đơn giản, và nó vẫn đang phổ biến trong các thống kê Bayes thực dụng. Nhưng nó đang nhanh chóng bị thay thế bởi các thuật toán khác.

### 9.2.1 Phương pháp lấy mẫu Gibbs

Thuật toán Metropolis hoạt động bất cứ khi nào xác suất đề cử một bước nhảy từ B đến A là bằng với xác suất đề cử A từ B, khi phân phối đề xuất là đối xứng. Có một phương pháp tổng quát hơn, được gọi là MetropolisHastings, cho phép các đề xuất không đối xứng. Điều này có nghĩa là, trong bối cảnh của cổ tích Vua Markov, rằng đồng xu của nhà vua bị sai lệch để dẫn ông theo chiều kim đồng hồ trung bình.

Tại sao chúng tôi muốn một thuật toán cho phép các đề xuất không đối xứng? Một lý do là nó giúp dễ dàng xử lý các tham số, chẳng hạn như độ lệch chuẩn, có ranh giới bằng 0. Tuy nhiên, một lý do tốt hơn là nó cho phép chúng ta tạo ra các đề xuất thông minh để khám phá phân phối posterior hiệu quả hơn. “Hiệu quả hơn” ở đây, ý tôi là chúng ta có thể có được một hình ảnh tốt như nhau về phân phối posterior với ít bước hơn.

Cách phổ biến nhất để tạo ra các đề xuất hiểu biết là một kỹ thuật được gọi là lấy mẫu Gibbs. Lấy mẫu Gibbs là một biến thể của thuật toán Metropolis-Hastings sử dụng các đề xuất thông minh và do đó hiệu quả hơn. “Hiệu quả” ở đây, ý tôi là bạn có thể ước tính tốt posterior từ việc lấy mẫu Gibbs với mẫu ít hơn so với cách tiếp cận Metropolis. Sự cải thiện nảy sinh ra từ các *đề xuất thích ứng*, trong đó việc phân phối các giá trị tham số được đề xuất sẽ tự điều chỉnh một cách thông minh, tùy thuộc vào các giá trị tham số tại thời điểm hiện tại.

Cách lấy mẫu Gibbs tính toán các đề xuất thích ứng này phụ thuộc vào việc sử dụng các kết hợp cụ thể của các phân phối prior và likelihood được gọi là các *cặp liên hợp (conjugate pair)*. Các cặp liên hợp có các giải pháp bằng phân tích cho phân phối posterior của một tham số riêng lẻ. Và những giải pháp này là những gì cho phép phương pháp lấy mẫu Gibbs thực hiện các bước nhảy thông minh xung quanh phân phôis posterior kết hợp của tất cả các tham số.

Trong thực tế, lấy mẫu Gibbs có thể rất hiệu quả và nó là cơ sở của phần mềm fit mô hình Bayes phổ biến như `BUGS` (Bayesian inference Using Gibbs Sampling) và `JAGS` (Just Another Gibbs Sampler). Trong các phần mềm này, bạn biên soạn mô hình thống kê của mình bằng cách sử dụng các định nghĩa rất giống với những gì bạn đã làm cho đến nay trong cuốn sách này. Phần mềm tự động hóa phần còn lại, với khả năng tốt nhất của nó.

### 9.2.2 Vấn đề nhiều chiều

Nhưng có một số hạn chế nghiêm trọng đối với lấy mẫu Gibbs. Đầu tiên, có thể bạn không muốn sử dụng các prior liên hợp. Một số prior liên hợp thực sự có hình dạng bị vấn đề, một khi bạn bắt đầu xây dựng mô hình đa tầng và cần prior cho toàn bộ ma trận hiệp phương sai. Đây sẽ là điều cần thảo luận khi chúng ta đến với Chương 14.

Thứ hai, khi các mô hình trở nên phức tạp hơn và chứa hàng trăm hoặc hàng nghìn hoặc hàng chục nghìn tham số, cả lấy mẫu Metropolis và Gibbs đều trở nên kém hiệu quả một cách đáng kinh ngạc. Lý do là chúng có xu hướng bị mắc kẹt ở các vùng nhỏ của posterior trong một thời gian dài. Số lượng tham số số cao không phải là vấn đề khó như vấn đề mà mô hình có nhiều tham số gần như luôn có các vùng mà tương quan cao trong posterior. Điều này có nghĩa là hai hoặc nhiều tham số có tương quan cao với nhau trong các mẫu posterior. Bạn đã từng thấy điều này trước đây với ví dụ về hai chân trong Chương 6. Tại sao đây là một vấn đề? Bởi vì tương quan cao có nghĩa là có một dải hẹp của các kết hợp có xác suất cao, và cả Metropolis và Gibbs đều đưa ra quá nhiều đề xuất ngớ ngẩn về nơi cần bước tiếp theo. Vì vậy, chúng có thể bị đứng yên.

Một bức tranh sẽ giúp làm rõ hơn điều này. [**HÌNH 9.3**](#f3) cho thấy một thuật toán Metropolis thông thường đang cố gắng khám phá posterior 2 chiều với mối tương quan âm mạnh là -0,9. Vùng của các giá trị tham số xác suất cao tạo thành một thung lũng hẹp. Bây giờ hãy tập trung vào biểu đồ bên trái. Chuỗi bắt đầu ở phía trên bên trái của thung lũng. Điểm đầy là đề xuất được chấp nhận. Điểm mở là những đề xuất bị từ chối. Các đề xuất được tạo ra bằng cách thêm nhiễu Gaussian ngẫu nhiên vào mỗi tham số, sử dụng độ lệch chuẩn 0,01, *chiều dài bước*. 50 đề xuất được hiển thị. Tỷ lệ chấp nhận chỉ là 60%, bởi vì khi thung lũng hẹp như thế này, các đề xuất có thể dễ dàng rơi ra ngoài nó. Nhưng dây chuyền vẫn di chuyển được từ từ xuống thung lũng. Nó di chuyển chậm, bởi vì ngay cả khi một đề xuất được chấp nhận, nó vẫn gần với điểm trước đó.

<a name="f3"></a>![](/assets/images/fig 9-3.svg)
<details class="fig"><summary>Hình 9.3: Chuỗi Metropolis dưới sự tương quan cao. Điểm màu xanh là đề xuất được chấp nhận. Điểm màu đỏ là bị từ chối. Cả hai biểu đồ cho thấy 50 đề xuất dưới chiều dài bước của phần phối đề xuất khác nhau. Trái: Với bước nhỏ, chuỗi di chuyển rất chậm xuống thung lũng. Nó từ chối 30% các đề xuất trong tiến trình, bởi vì đa số các đề xuất đề ngu ngốc. Phải: Với chiều dài bước lớn hơn, chuỗi đi nhanh hơn, nhưng bây giờ nó từ chối 50% các đề xuất, bởi vì chúng thường ngu ngốc hơn. Ở nhiều chiều, hầu như bất khả thi để tuỳ chỉnh Metropolis hoặc Gibbs để hiệu quả hơn.</summary>
{% highlight python %}_, axes = plt.subplots(1,2,figsize=(9,4), subplot_kw={'xlim':(-1.5,1.5), 'ylim':(-1.5,1.5)})
cov = [[0.5,-.45],[-.45,0.5]]
a = numpyro.sample('a', dist.MultivariateNormal(0, cov), rng_key=random.PRNGKey(1), sample_shape=(1000,))
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
for ax, stepsize in zip(axes, [0.1,0.25]):
    nums = 50
    positions = jnp.zeros([nums,2])
    current = jnp.array([-0.1, -0.1])
    counter = 0 
    accept = jnp.zeros(nums)
    def body_fn(i, val):
        positions, current, counter, accept = val
        positions = ops.index_update(positions, i, current)
        new_x = dist.Normal(current[0], stepsize).sample(random.fold_in(random.PRNGKey(0), i))
        new_y = dist.Normal(current[1], stepsize).sample(random.fold_in(random.PRNGKey(1), i))
        proposal = jnp.array([new_x,new_y])
        prev = jnp.exp(dist.MultivariateNormal(0, cov).log_prob(current))
        new_prob = jnp.exp(dist.MultivariateNormal(0, cov).log_prob(proposal))
        ratio = new_prob/prev
        prob_move = jnp.min(jnp.array([1, ratio]))
        unif = dist.Uniform().sample(random.fold_in(random.PRNGKey(2), i))
        counter += (unif < prob_move)
        accept = ops.index_update(accept, i, (unif < prob_move))
        current = jnp.where((unif < prob_move), proposal, current)
        return positions, current, counter, accept
    positions, current,counter, accept = lax.fori_loop(0, nums, body_fn, (positions, current,counter, accept))
    sns.kdeplot(x=a[:,0], y=a[:,1], ax=ax, alpha=0.4)
    sns.scatterplot(
        x=positions[:,0][accept.astype(bool)],
        y=positions[:,1][accept.astype(bool)],
        color='C0',ax=ax)
    sns.scatterplot(
        x=positions[:,0][~accept.astype(bool)],
        y=positions[:,1][~accept.astype(bool)],
        color='C1',ax=ax)
    ax.set(title=f"chiều dài bước: {stepsize}\ntỉ lệ chấp nhận: {counter/nums:.02f}",
           xlabel='a1', ylabel='a2'){% endhighlight %}</details>


Điều gì xảy ra sau đó nếu chúng ta tăng chiều dài bước, cho các đề xuất xa hơn? Bây giờ nhìn bên phải trong [**HÌNH 9.3**](#f3). Hiện chỉ có 30% đề xuất được chấp nhận. Chiều dài bước lớn hơn có nghĩa là nhiều đề xuất ngớ ngẩn hơn bên ngoài thung lũng. Tuy nhiên, các đề xuất được chấp nhận di chuyển nhanh hơn dọc theo chiều dài của thung lũng. Trên thực tế, rất khó để giành được sự đánh đổi này. Cả Metropolis và Gibbs đều gặp khó khăn như vậy, bởi vì các đề xuất của họ không đủ hiểu biết về hình dạng toàn cầu của posterior. Họ không biết họ đang đi đâu.

Ví dụ về sự tương quan cao đã minh họa vấn đề này. Nhưng vấn đề thực tế nghiêm trọng hơn và thú vị hơn. Bất kỳ các tiếp cận chuỗi Markov nào lấy mẫu các tham số riêng lẻ trong các bước riêng lẻ sẽ gặp khó khăn, một khi số lượng tham số tăng đủ lớn. Lý do là vì **NỒNG ĐỘ CỦA ĐO LƯỜNG**. Đây là một cái tên khó hiểu cho sự thật kinh ngạc là phần lớn khối lượng xác suất của phân phối nhiều chiều luôn nằm rất xa so với điểm mode của phân phối. Thật khó để hình dung. Chúng ta không thể nhìn thấy 100 chiều, vào hầu hết các ngày. Nhưng nếu chúng ta nghĩ bằng phiên bản 2D và 3D, chúng ta có thể hiểu được hiện tượng cơ bản. Trong hai chiều, phân bố Gaussian là một ngọn đồi. Điểm cao nhất là ở giữa, là điểm mode. Nhưng nếu chúng ta tưởng tượng ngọn đồi này đầy bùn đất — ngoài đất ra còn gì nữa? - thì chúng ta có thể hỏi: Phần lớn bùn đất nằm ở đâu? Khi chúng ta di chuyển khỏi đỉnh theo bất kỳ hướng nào, độ cao sẽ giảm xuống, do đó sẽ có ít bụi bẩn trực tiếp dưới chân chúng ta hơn. Nhưng trong một vòng tròn quanh ngọn đồi ở cùng một khoảng cách, có nhiều đất hơn ở đỉnh. Diện tích tăng lên khi chúng ta di chuyển ra khỏi đỉnh, mặc dù độ cao giảm xuống. Vì vậy, tổng số lượng bùn đất, hay xác suất, tăng lên khi chúng ta di chuyển ra khỏi đỉnh. Cuối cùng, tổng số lượng bùn đất (xác suất) lại giảm xuống, khi ngọn đồi dốc xuống bằng không. Vì vậy, tại một số khoảng cách tính từ đỉnh, bùn đất (khối lượng xác suất) là cực đại. Trong không gian ba chiều, nó không phải là một ngọn đồi, mà bây giờ là một hình cầu mờ ảo. Hình cầu dày đặc nhất ở lõi, là "đỉnh" của nó. Nhưng một lần nữa thể tích lại tăng lên khi chúng ta di chuyển ra khỏi lõi. Vì vậy, có nhiều tổng thể hình cầu hơn trong lớp vỏ xung quanh lõi.

Quay lại với suy nghĩ về phân phối xác suất, tất cả điều này có nghĩa là sự kết hợp của các giá trị tham số tối đa hóa xác suất posterior, điểm mode, không thực sự nằm trong vùng giá trị tham số có tính phù hợp cao. Điều này có nghĩa là khi chúng ta lấy mẫu đúng cách từ phân phối nhiều chiều, chúng ta sẽ không nhận được bất kỳ điểm nào gần điểm mode. Bạn có thể tự chứng minh điều này rất dễ dàng. Chỉ cần lấy mẫu ngẫu nhiên từ phân phối nhiều chiều — 10 chiều là đủ — và vẽ biểu đồ khoảng cách ly tâm của các điểm. Dưới đây là code để thực hiện việc này:

<b>code 9.4</b>
```python
D = 10
T = int(1e3)
Y = dist.MultivariateNormal(jnp.repeat(0, D), jnp.identity(D)).sample(
    random.PRNGKey(0), (T,)
)
rad_dist = lambda Y: jnp.sqrt(jnp.sum(Y ** 2))
Rd = lax.map(lambda i: rad_dist(Y[i]), jnp.arange(T))
az.plot_kde(Rd, bw=0.18)
```

<a name="f4"></a>![](/assets/images/fig 9-4.svg)
<details class="fig"><summary>Hình 9.4: Nồng độ đo lường và lời nguyền của nhiều chiều. Trục hoành là khoảng cách ly tâm từ điểm mode trong không gian tham số. mỗi mật độ là một mẫu gồm 1000 điểm. Con số trên mỗi mật độ là số chiều. Khi số lượng tham số tăng lên, thì điểm mode càng xa giá trị bạn muốn lấy mẫu.</summary>
{% highlight python %}T = int(1e3)
Y = lambda D: dist.MultivariateNormal(jnp.repeat(0, D), jnp.identity(D)).sample(
    random.PRNGKey(0), (T,)
)
rad_dist = lambda Y: jnp.sqrt(jnp.sum(Y ** 2))
Rd = lambda D: lax.map(lambda i: rad_dist(Y(D)[i]), jnp.arange(T))
for i,t in zip([1,10,100,1000], [0,2,8.5,30]):
    az.plot_kde(Rd(i),bw=0.1)
    plt.annotate(i,(t,0.7) ){% endhighlight %}</details>

Tôi thể hiện mật độ này, cùng với mật độ tương ứng với phân phối có 1, 100, và 1000 chiều, ở [**HÌNH 9.4**](#f4). Trục hoành ở đây là khoảng cách ly tâm của các điểm từ điểm mode. Cho nên giá trị 0 là đỉnh của xác suất. Bạn có thể thấy rằng phân phối Gaussian thông thường với chỉ 1 chiều, bên trái, lấy phần lớn các mẫu bên cạnh đỉnh này, như bạn mong đợi. Nhưng với 10 chiều, đã không có mẫu nào bên cạnh đỉnh ở zero. Với 100 chiều, chúng ta đã đi xa khỏi đỉnh. Và với 1000 chiều, nó càng xa hơn. Số mẫu nằm trong một không gian rất chật, nhiều chiều cách xa điểm mode. Không gian này tạo ra những con đường rất trắc trở để công cụ lấy mẫu tiếp cận vào.

Đó là tại sao chúng ta cần thuật toán MCMC mà tập trung trên toàn bộ posterior cùng lúc, thay vì một hay vài chiều cùng lúc như Metropolis và Gibbs. Nếu không chúng ta sẽ gặp khó khăn trong một vùng chật hẹp, rất cong của không gian tham số.

## <center>9.3 Hamiltonian Monte Carlo</center><a name="a3"></a>


## <center>9.4 HMC đơn giản</center><a name="a4"></a>
## <center>9.5 Chăm sóc và nuôi dưỡng chuỗi Markov</center><a name="a5"></a>
## <center>9.6 Tổng kết</center><a name="a6"></a>


<a name="f1"></a>![](/assets/images/fig -.svg)
<details class="fig"><summary></summary>
{% highlight python %}{% endhighlight %}</details>

---

<details><summary>Endnotes</summary>
<ol class='endnotes'>
    <li></li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
</details>