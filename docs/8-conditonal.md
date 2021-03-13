---
title: "Chapter 8: Conditional Manatees"
description: "Chương 8: Những con lợn biển có điều kiện"
---

- [8.1 Xây dựng sự tương tác](#a1)
- [8.2 Tính đối xứng của sự tương tác](#a2)
- [8.3 Sự tương tác liên tục](#a3)
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
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Lợn biển (manatee - *Trichechus manatus*) là một động vật hữu nhũ chậm chạp, sống dưới vùng nước ấm và nông. Lợn biển không có thiên dịch tự nhiên, nhưng chúng có chia sẻ vùng nước với những chiếc thuyền máy. Và thuyền máu có cánh quạt. Khi những con lợn biển liên quan với loài voi và có lớp da rất dày, cánh quạt có thể giết chúng. Phần lớn các con lợn biển trưởng thành đều có sẹo từ va chạm với thuyền ([**HÌNH 8.1, trên**](#f1))<sup><a name="r135" href="#135">135</a></sup>

<a name="f1"></a>![](/assets/images/fig 8-1.png)
<details class="fig"><summary>Hình 8.1: TRÊN: Các vết sẹo trên lưng của 5 con lợn biển Florida. Các dãy trầy xước, như trên các cá thể Africa và Flash, là dấu hiệu của vết thương do cánh quạt. DƯỚI: Ba ví dụ thiệt hại trên những chiếc máy bay thả bom A.W.38 sau nhiệm vụ.</summary></details>

Armstrong Whitworth A.W.38 Whitley là máy bay thả bom tiền tuyến của Lực Lượng Hàng Không Hoàng Gia. Vào Thế Chiến II, A.W.38 mang bom và thả vào địa phận của Đức. Không giống như lợn biển, A.W.38 có thiên địch tự nhiên hung tợn: pháo phản không. Nhiều máy bay không bao giờ trở về từ nhiệm vụ. Và thiệt hại trên những chiếc còn sống sót đã chứng minh điều đó ([**HÌNH 8.1, dưới**](#f1))

Lợn biển có giống máy bay A.W.38 không? Trong cả hai trường hợp - vết trầy do máy quạt của lợn biển và các lỗ thủng do đạn của máy bay - chúng ta muốn làm gì đó để cải thiện khả năng giúp lợn biển và máy bay sống sót. Nhiều người quan sát nghĩ rằng giúp lợn biển và máy bay nghĩa là giảm loại tổn thương mà chúng ta nhìn thấy trên chúng. Với lợn biển, nó nghĩa là cần thêm lớp bảo vệ trên cánh quạt (của thuyền, không phải của lợn biển). Với máy bay, nó nghĩa là cần thêm giáp ở những bộ phận máy bay có nhiều tổn thương nhất.

Trong cả hai trường hợp, bằng chứng đã gây hiểu sai. Cánh quạt không gây ra những tổn thương và cái chết cho lợn biển. Thực vậy tử thiết trên chúng khẳng định sự va chạm với các bộ phận tù của chiếc thuyền nhưng mái chéo, cho tổn thương nhiều hơn. Tương tự, tăng giáp ở những thành phần tổn thương không cải thiện cho các máy bay thả bom. Thay vào đó, nâng cấp máy bay thả bom A.W.38 nghĩa là nâng giáp các bộ phận *không bị tổn thương*.<sup><a name="r136" href="#136">136</a></sup> Ví dụ từ những con lợn biển và máy bay thả bom còn sống sót là gây hiểu sai, bởi vì nó *đặt điều kiện* trên sống còn. Lợn biển và máy bay bị tiêu diệt trông sẽ khác. Một con lợn biển bị va chạm với máu chèo sẽ ít khả năng sống sót hơn nhưng con bị cánh quạt trợt qua. Nên giữa các con sống sót, vết trầy do cánh quạt là thường gặp. Tương tự, những máy bay trở về được không có vết thương rõ ràng trên động cơ và buồng lái. Họ may mắn. Máy bay thả bom không về nhà được thì ít như vậy hơn. Để có được đáp án chính xác, trong hai trường hợp trên, chúng ta cần phải nhận ra loại tổn thương nhìn thấy này là được đặt điều kiện trên sống còn.

**ĐẶT ĐIỀU KIỆN** là một trong những nguyên tắc quan trọng nhất trong suy luận thống kê. Data, như các vết xước lợn biển và tổn thương máy bay, được đặt điều kiện trên cách chúng được đưa vào mẫu. Phân phối posterior được đặt điều kiện trên data. Tất cả mọi suy luận bằng mô hình được đặt điều kiện trên mô hình. Mọi suy luận được đặt điều kiện trên một thứ gì đó, cho dù chúng ta có nhận ra hay không.

Và một phần lớn sức mạnh của mô hình thống kê đến từ việc tạo ra các thiết bị cho phép xác suất được đặt điều kiện trên các khía cạnh của từng trường hợp. Mô hình tuyến tính mà bạn quen thuộc mà thiết bị thô sơ cho phép mọi kết cục $y_i$ được đặt điều kiện trên một tập các biến dự đoán cho mỗi trường hợp $i$. Giống như các epicycle của mô hình Ptolemaic và Kopernikan (Chương 4 và 7), mô hình tuyến tính cho chúng ta phương pháp mô tả khả năng đặt điều kiện.

Mô hình tuyến tính đơn giản thông thường không có khả năng cung cấp đủ các phép đặt điều kiện, tuy nhiên . Mọi mô hình đến bây giờ trong sách này giả định mỗi biến dự đoán có mối quan hệ độc lập với trung bình của kết cục. Nhưng nếu chúng ta muốn cho phép mối quan hệ được đặt điều kiện thì sao? Ví dụ, trong data sữa các loài khỉ từ các chương trước, giả sử quan hệ giữa năng lượng sữa và kích thước não thay đổi theo nhóm loài (khỉ, vượn, tinh tinh). Điều này giống như nói rằng ảnh hưởng của kích thước não trên năng lượng sữa được đặt điều kiện trên nhóm loài. Mô hình tuyến tính từ chương trước không giải quyết được câu hỏi này.

Để mô hình hoá điều kiện sâu hơn - khi mức độ quan trọng của một biến phụ thuộc vào một biến dự đoán khác - chúng ta cần **SỰ TƯƠNG TÁC (INTERACTION)** (cũng được biết **SỰ ĐIỀU TIẾT - MODERATION**). Tương tác là một loại đặt điều kiện, một cách để cho phép tham số (thực ra là phân phối posterior) được đặt điều kiện trên các khía cạnh xa hơn của data. Loại tương tác đơn giản nhất, tương tác tuyến tính, được xây dựng bằng cách mở rộng chiến thuật mô hình tuyến tính vào tham số trong mô hình tuyến tính. Cho nên nó đồng nghĩa với việc thay thế epicycle trên epicycle trong mô hình Ptolemaic và Kopernikan. Nó mang tính mô tả, nhưng rất mạnh.

Tổng quát hơn, tương tác là trung tâm của đa số mô hình thống kê đằng sau thế giới ấm áp của kết cục Gaussian và mô hình tuyến tính của trung bình. Trong mô hình tuyến tính tổng quát (GLM, Chương 10 và sau đó), ngay cả khi người ta không định nghĩa rõ ràng các biến là tương tác, chúng vẫn tương tác ở một mức độ nào đó. Mô hình đa tầng cũng cho hiệu ứng tương tự. Mô hình đa tầng thông thường là một mô hình tương tác khổng lồ, trong đó các giá trị ước lượng (intercept và slope) được đặt điều kiện cho cụm (người, loài, làng, thành phố, vũ trụ) trong data. Hiệu ứng tương tác đa tầng là phức tạp. Chúng không chỉ cho phép ảnh hưởng của biến dự đoán thay đổi phụ thuộc vào một biến khác, mà còn ước lượng khía cạnh của *phân phối* của những thay đổi đó. Điều này nghe có vẻ thiên tài, hoặc điên rồ, hoặc cả hai. Cho dù thế nào, bạn không thể có sức mạnh của mô hình tầng mà không có nó.

Mô hình cho phép tương tác phức tạp thì dễ fit vào data. Nhưng chúng cũng được cho là khó hiểu hơn. Và nên tôi dành chương này nói về các hiệu ứng tương tác đơn giản: làm sao để định nghĩa, diễn giải, và minh hoạ chúng. Chương này bắt đầu bằng một trường hợp tương tác giữa một biến phân nhóm và một biến liên tục. Trong bối cảnh này, rất dễ để nhận ra dạng giả thuyết cho phép sự tương tác. Rồi sau đó chương này nói về tương tác phức tạp hơn giữa các biến dự đoán liên tục. Nó khó hơn. Trong tất cả các phần của chương này, dự đoán của mô hình được minh hoạ, trung bình hoá trên tính bất định trong tham số.

Sự tương tác là bình thường, nhưng chúng không dễ. Hi vọng là chương này là tạo một nền tảng vững chắc cho việc diễn giải mô hình tuyến tính tổng quát và mô hình đa tầng trong các chương sau.

<div class="alert alert-info">
<p><strong>Minh tinh thống kê, Abraham Wald.</strong> Câu chuyện máy bay thả bom trong Thế Chiến II là tác phẩm của Abraham Wald (1902-1950). Wald sinh ra ở nơi mà bây giờ gọi là Romania, nhưng di cư sang Mỹ sau khi Nazi xâm chiếm nước Áo. Wald đã cống hiến rất nhiều trong cuộc đời ngắn ngủi của ông. Có lẽ công tình liên quan nhất đến tài liệu này, là Wald đã chứng mình rằng với nhiều loại quy luật để quyết định theo thống kê, luôn luôn tồn tại một quy luật Bayes chí ít tốt bằng nhiều quy luật non-Bayes. Wald đã chứng minh điều này, một cách xuất sắc, bắt đầu với các tiền đề non-Bayes, và nên phe anti-Bayes không thể mặc kệ nó nữa. Công trình này được tóm tắt trong sách 1950 của Wald, được phát hành chỉ trước khi ông mất.<sup><a name="r137" href="#137">137</a></sup> Wald chết khi quá trẻ, từ một vụ rơi máy bay khi tham quan Ấn Độ.</p></div>

## <center>8.1 Xây dựng sự tương tác</center><a name="a1"></a>

Châu Phi rất đặc biệt. Lục địa lớn thứ hai, đa dạng về văn hoá và di truyền. Châu Phi có 3 tỉ người ít hơn so với Châu Á, nhưng nó lại có nhiều ngôn ngữ giao tiếp. Châu Phi đa dạng di truyền và đa số các biến thể di truyền ngoài Châu Phi là một phần nhỏ của biến thể trong Châu Phi. Châu Phi cũng đặc biệt về địa hình, theo một cách kỳ lạ: Địa hình xấu thường liên quan quan đến kinh tế xấu ngoài Châu Phi, nhưng kinh tế ở Châu Phi lại thực ra hưởng lợi từ địa hình xấu.

Để hiểu sự kỳ lạ này, hãy nhìn vào hồi quy của mức độ gồ ghề địa hình - một loại địa hình xấu - đối với hiệu năng kinh tế (log GDP<sup><a name="r138" href="#138">138</a></sup> trên đầu người vào năm 2000), cả trong và ngoài Châu Phi ([**HÌNH 8.2**](#f2)). Biến số `rugged` là Chỉ Số Gồ Ghề Địa Hình<sup><a name="r139" href="#139">139</a></sup> dùng để định lượng tính hỗn tạp cấu trúc của một vùng đất. Biến kết cục ở đây là logarith của tổng sản phẩm nội địa (gross domestic product - GDP) bình quân đầu người, từ năm 2000, `rgdppc_2000`. Chúng tôi sử dụng logarith của nó, bởi vì logarith của GDP là *mức độ* của GDP. Bởi vì sự giàu có tạo ra sự giàu có, nó có xu hướng tăng luỹ thừa liên quan với bất cứ thứ gì làm nó tăng. Nó giống như nói rằng khoảng cách tuyệt đối trong sự giàu có tăng ngày càng lớn, khi đất nước giàu có hơn. Cho nên khi chúng ta làm việc với logarith, chúng ta đang làm việc trên thang mức độ được chia đều hơn. Cho dù thế nào, hãy nhớ rằng chuyển đổi log không làm mất thông tin. Nó chỉ thay đổi những giả định của mô hình về hình dáng của quan hệ giữa các biến. Trong trường hợp này, GDP thô không quan hệ tuyến tính với bất cứ thứ gì, bởi vì hình dạng luỹ thừa của nó. Nhưng log GPD lại quan hệ tuyến tính với rất nhiều thứ.

<a name="f2"></a>![](/assets/images/fig 8-2.svg)
<details class="fig"><summary>Hình 8.2: Hồi quy tuyến tính khác nhau giữa trong và ngoài Châu Phi, giữa log GDP và mức độ gồ ghề địa hình. Slope dương trong Châu Phi, nhưng là âm khi ở ngoài. Làm sao chúng ta có thể tái hiện sự đảo ngược của slope, bằng data kết hợp?</summary>
{% highlight python %}url = r'https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv'
df = pd.read_csv(url+"?raw=true",sep=';')
df['rug_std']= df['rugged'].pipe(lambda x : (x-x.min())/(x.max()-x.min()))
df['log_gdp']= np.log(df['rgdppc_2000'])
df = df.dropna(subset=['log_gdp'])
df['log_gdp_p'] = df['log_gdp']/(df['log_gdp'].mean())
fig, axs = plt.subplots(1,2,figsize=(10,5))
sns.regplot(x=df['rug_std'][df['cont_africa']==1],
            y=df['log_gdp_p'][df['cont_africa']==1],ax=axs[0])
sns.regplot(x=df['rug_std'][df['cont_africa']==0],
            y=df['log_gdp_p'][df['cont_africa']==0],ax=axs[1])
for ax in axs:
    ax.set(xlabel="độ gồ ghề (chuẩn hoá)", ylabel="log GDP (tỉ lệ với trung bình)")
axs[0].set(title="Quốc gia Châu Phi")
axs[1].set(title="Quốc gia không Châu Phi")
for c in ['Lesotho', 'Seychelles']:
    entry = df[df['country']==c]
    axs[0].annotate(c, (entry['rug_std'],entry['log_gdp_p']))
for c in ['Switzerland', 'Tajikistan']:
    entry = df[df['country']==c]
    axs[1].annotate(c, (entry['rug_std'],entry['log_gdp_p'])){% endhighlight %}</details>

Chuyện gì đã xảy ra trong hình này? Đồng ý là độ gồ ghề liên quan với những nước nghèo, ở phần lớn thế giới. Địa hình gồ ghề nghĩa là giao thông khó khăn hơn. Tức là tiếp cận thị trường bị cản trở. Tức là giảm tổng sản phẩm nội địa. Cho nên quan hệ trái chiều trong Châu Phi là khó hiểu. Tại sao địa hình khó khăn lại liên quan đến GPD bình quân đầu người cao hơn?

Nếu quan hệ này là nhân quả, thì có thể bởi vì những vùng đất gồ ghề của Châu Phi được bảo vệ tránh khỏi những vụ mua bán nô lệ ở Đại Tây Dương và Ấn Độ Dương. Những người mua bán nô lệ thường nhắm vào những địa bàn dễ tấn công, với những đường đi dễ dàng ra biển. Những vùng này thường chịu đựng việc mua bán nô lệ tiếp tục bị thoái hoá kinh tế, sau khi thị trường nô lệ giảm đi. Và mức độ gồ ghề là tương quan với những đặc điểm địa hình khác, như đường bờ biển, nó cũng ảnh hưởng kinh tế. Cho nên rất khó để chắc chắn những gì đang xảy ra.

Giả thuyết nhân quả, dưới dạng DAG, có thể là (xem thêm cuối phần này):

![](/assets/images/dag 8-1.svg)

Trong đó $R$ là độ gồ ghề địa hình, $G$ là GDP, $C$ là lục địa, và $U$ là một tập nhiễu không quan sát được (như khoảng cách đến bờ biển). Hãy tạm mặc kệ $U$. Bạn sẽ xem xét những yếu tố nhiễu khác trong phần thực hành. Thay vào đó tập trung vào những gợi ý mà $R$ và $C$ đều ảnh hưởng $G$. Điều này có nghĩa là nguồn ảnh hưởng độc lập hoặc có thể chúng tương tác (một biến điều hoà ảnh hưởng của biến còn lại). DAG này không thể hiện sự tương tác. Bởi vì DAG không cụ thể cách mà hai biến kết hợp ảnh hưởng lên biến khác. DAG trên chỉ gợi ý rằng có một hàm sử dụng cả $R$ và $C$ để tạo ra $G$. Trong ký hiệu điển hình $G=f(R,C)$.

Cho nên chúng ta cần một cách tiếp cận thống kê để đánh giá những định đề khác nhau cho $f(R,C)$. Làm sao tạo mô hình cho khả năng đặt điều kiện trong [**HÌNH 8.2**](#f2)? Chúng ta có thể đánh lừa bằng cách chia data thành hai DataFrame, một cho Châu Phi và một cho những lục địa khác. Nhưng nó không phải ý tưởng tốt để chia data như vậy. Đây là bốn lý do.

Thứ nhất, thông thường có vài tham số, như $\sigma$, mà mô hình cho rằng không phụ thuộc vào lục địa nào. Bằng cách chia thành các bảng, bạn đã gây tổn hại đến độ chính xác của ước lượng cho những tham số này, bởi vì bạn đã làm cho ước lượng kém chính xác hai lần thay vì gồm tất cả những bằng chứng vào một trị số ước lượng. Do đó, bạn đã vô tình giả định rằng phương sai khác nhau giữa Châu Phi và các quốc giá không phải Châu Phi khác. Bây giờ, không có gì sai với giả định đó. Nhưng bạn muốn tránh những giả định vô ý.

Thứ hai, để đạt được mệnh đề xác suất của biến số mà bạn dùng để tách data ra, `cont_africa` trong trường hợp này, bạn cần thêm nó vào mô hình. Nếu không, bạn sẽ có một kết luận thống kê yếu. Có hay không tính bất định của một giá trị dự đoán phân biệt giữa Châu Phi và quốc gia không phải Châu Phi? Dĩ nhiên là có. Nếu bạn phân tích toàn bộ data trong một mô hình duy nhất, bạn sẽ không dễ định lượng tính bất định đó. Nếu bạn chỉ để phân phối posterior làm việc cho bạn, bạn sẽ có một ước lượng hữu ích về tính bất định đó.

Thứ ba, chúng ta muốn sử dụng tiêu chuẩn thông tin hoặc một phương thức khác để so sánh mô hình. Để so sánh một mô hình mà xem tất cả các lục địa theo cùng một cách với một mô hình cho phép slope khác nhau trong lục địa khác nhau, chúng ta cần mô hình sử dụng tất cả cùng một data (như đã giải thích trong Chương 7). Điều này có nghĩa chúng ta không thể tách data thành hai mô hình riêng biệt. Chúng ta cần phải để cho một mô hình duy nhất tách data nội tại.

Thứ tư, khi bạn bắt đầu sử dụng mô hình đa tầng (Chương 13), bạn sẽ thấy rằng có nhiều lợi ích thì mượn thông tin xuyên suốt phân nhóm như "Châu Phi" và "không Châu Phi". Điều này đúng đặc biệt khi cỡ mẫu thay đổi giữa các phân nhóm, và nguy cơ overfitting là cao hơn trong một vài nhóm. Nói cách khác, những gì chúng ta học về mức độ gồ ghề ngoài Châu Phi nên có vài hiệu ứng lên ước lượng trong Châu Phi, và ngược lại. Mô hình đa tầng (Chương 13) mượn thông tin bằng cách này, để cải thiện ước lượng cho mọi phân nhóm. Khi chúng ta tách data, việc mượn này là không khả thi.

<div class="alert alert-dark">
<p><strong>Nhân quả không dễ dàng.</strong> DAG mức đồ gồ ghề địa hình trong phần trước là dễ. Nhưng sự thật không đơn giản như vậy. Lục địa không phải nguồn quan tâm chính. Có thể có sự phơi nhiễm trong lịch sử theo lý thuyết của chủ nghĩa thực dân và mua bán nô lệ đã ảnh hưởng dai dẳng đến hiệu suất kinh tế. Những đặc tính của địa hình, như độ gồ ghề, theo nhân quả giảm những yếu tố lịch sử đó có thể ảnh hưởng gián tiếp vào kinh tế. Như vậy:</p>
<img src="./assets/images/dag 8-2.svg">
<p>$H$ là những yếu tố lịch sử như phơi nhiễm mua bán nô lệ. Tổng hiệu ứng nhân quả của $R$ bao gồm con đường trực tiếp $R \to G$ (được giả định là luôn luôn âm) và con đường gián tiếp $R \to H \to G$. Con đường thứ hai là con đường mà cùng biến thiên với lục địa $C$, bởi vì $H$ là quan hệ mạnh với $C$. Chú ý rằng biến nhiễu $U$ có thể ảnh hưởng bất kỳ biến nào (ngoại trừ $C$). Ví dụ nếu khoảng cách đến bờ biến là biến thực sự ảnh hưởng $H$ trong quá khứ, không phải độ gồ ghề địa hình, thì quan hệ giữa độ gồ ghề địa hình với GDP là không nhân quả. Data gồm một lượng lớn yếu tố nhiễu tiềm nằng mà bạn nên xem xét. Hệ thống tự nhiên như thế là phức tạp kinh khủng.</p></div>

### 8.1.1 Tạo mô hình gồ ghề

Hãy xem cách tái diễn sự đảo chiều slope này, chỉ trong một mô hình. Chúng ta bắt đầu bằng fit một mô hình duy nhất với mọi data, bỏ qua lục địa. Nó cho phép chúng ta suy nghĩ về cấu trúc mô hình và prior trước khi đối mặt với ác quỷ tương tác. Để bắt đầu, tải data về và tiền xử lý:

<b>code 8.1</b>
```python
d = pd.read_csv(r'https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv?raw=true', sep=";")
# make log version of outcome
d["log_gdp"] = d["rgdppc_2000"].apply(math.log)
# extract countries with GDP data
dd = d[d["rgdppc_2000"].notnull()].copy()
# rescale variables
dd["log_gdp_std"] = dd.log_gdp / dd.log_gdp.mean()
dd["rugged_std"] = dd.rugged / dd.rugged.max()
```

Mỗi dòng trong data là một quốc gia, và nhiều cột trong đó là kinh tế, địa hình và đặc tính lịch sử.<sup><a name="r140" href="#140">140</a></sup> GDP thô và độ gồ ghề địa hình không có ý nghĩa lắm cho con người. Cho tên tôi đã chuẩn hoá các biến để thành đơn vị dễ sử dụng hơn. Chuẩn hoá như thường lệ bằng cách trừ trung bình và chia cho độ lệch chuẩn. Nó giúp các biến trở thành z-score. Chúng ta không muốn thực hiện nó ở đây, bởi vì độ gồ ghề bằng không là có ý nghĩa. Cho nên thay vì độ gồ ghề được chia cho giá trị quan sát lớn nhất. Nghĩa là nó được chuẩn hoá thành thang đo từ hoàn toàn phẳng (zero) thành tối đa ở mẫu là 1 (Lesotho, một nơi rất gồ ghề và xinh đẹp). Tương tự, log GDP đuọc chia cho giá trị trung bình. Nên nó được chỉnh lại thành thang đo tỉ lệ với trung bình quốc tế. 1 là trung bình, 0.8 là 80% của trung bình, và 1.1 và 10% hơn trung bình

Để xây dựng mô hình Bayes cho quan hệ này, chúng ta lần nữa sử dụng một khung địa tâm:

$$\begin{matrix}
\log(y_i) &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta(r_i - \bar{r})\\
\end{matrix}$$

Trong đó, $y_i$ là GDP quốc gia $i$, $r_i$ là độ gồ ghề địa hình của quốc gia $i$, và $\bar{r}$ là độ gồ ghề trung bình của toàn bộ mẫu. Giá trị của nó là 0.215 - hầu hết các quốc gia là không gồ ghề. Nhớ rằng sử dụng $\bar{r}$ chỉ làm nó dễ hơn khi gán prior vào intercept $\alpha$.

Cái khó khăn ở đây là khi chúng ta cụ thể hoá prior. Nếu bạn giống tôi, bạn không có nhiều thông tin khoa học về quan hệ phù hợp giữa log GDP và độ gồ ghề địa hình. Nhưng ngay khi chúng ta không biết nhiều về bối cảnh, cách thức đo đạc tự nó ràng buộc prior theo những cách hữu ích. Kết cục và biến dự đoán được chuẩn hoá sẽ làm điều này dễ dàng hơn. Xem xét intercept trước, $\alpha$, định nghĩa là log GDP khi độ gồ ghề ở trung bình mẫu. Cho nên nó phải gần bằng 1, bởi vì chúng ta chuẩn hoá kết cục để trung bình là 1. Hãy bắt đầu bằng việc đoán:

$$ \alpha \sim \text{Normal}(0,1)$$

Bây giờ với slope $\beta$. Nếu chúng ta đặt nó ở giữa zero, tức là không có sai lệch dương hoặc âm, thì nó hợp lý. Những với độ lệch chuẩn thì sao? Hãy bắt đầu bằng đoán con số 1:

$$ \beta \sim \text{Normal}(0,1)$$

Chúng ta sẽ đánh giá ước đoán này bằng cách mô phỏng phân phối dự đoán prior. Điều cuối cùng chúng ta cần là prior của $\sigma$. Hãy gán nó bằng một thứ gì đó rộng, $\sigma \sim \text{Exponential}(1)$. Trong thực hành cuối chương, tôi sẽ yêu cầu bạn đối đầu với prior này. Nhưng chúng ta hãy bỏ nó qua trong ví dụ này.

Sau cùng, chúng ta có được mô hình ứng cử cho data gồ ghề địa hình:

<b>code 8.2</b>
```python
def model(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_1, losses = svi.run(random.PRNGKey(0), 1000)
```

Chúng ta sẽ chưa nhìn vào dự đoán posterior, mà vào dự đoán prior. Hãy trích xuất prior và thể hiện những đường thẳng suy ra từ nó.

<b>code 8.3</b>
```python
predictive = Predictive(m8_1.model, num_samples=1000, return_sites=["a", "b", "sigma"])
prior = predictive(random.PRNGKey(7), rugged_std=0)
# set up the plot dimensions
plt.subplot(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="ruggedness", ylabel="log GDP")
plt.gca().axhline(dd.log_gdp_std.min(), ls="--")
plt.gca().axhline(dd.log_gdp_std.max(), ls="--")
# draw 50 lines from the prior
rugged_seq = jnp.linspace(-0.1, 1.1, num=30)
mu = Predictive(m8_1.model, prior, return_sites=["mu"])(
    random.PRNGKey(7), rugged_std=rugged_seq
)["mu"]
for i in range(50):
    plt.plot(rugged_seq, mu[i], "k", alpha=0.3)
```

<a name="f3"></a>![](/assets/images/fig 8-3.svg)
<details class="fig"><summary>Hình 8.3: Mô phỏng để tìm kiếm prior hợp lý cho ví dụ độ gồ ghề địa hình. Đường nằm ngang chỉ điểm các quan sát GDP tối đa và tối thiểu quan sát được. Trái: lần đoán đầu tiên với prior mơ hồ. Phải: Mô hình cải tiến với những prior phù hợp hơn.</summary>
{% highlight python %}def modela(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
def modelb(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
fig, axs = plt.subplots(1,2, figsize=(10,5))
for ax, model in zip(axs, [modela, modelb]):
    guide = AutoLaplaceApproximation(model)
    svi = SVI(
        model, guide, optim.Adam(0.1), Trace_ELBO(),
        rugged_std=dd.rugged_std.values,
        log_gdp_std=dd.log_gdp_std.values,
    )
    params, losses = svi.run(random.PRNGKey(0), 1000)
    predictive = Predictive(model, num_samples=1000, return_sites=["a", "b", "sigma"])
    prior = predictive(random.PRNGKey(7), rugged_std=0)
    ax.set(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="độ gồ ghề", ylabel="log GDP (tỉ lệ với trung bình)")
    ax.axhline(dd.log_gdp_std.min(), ls="--")
    ax.axhline(dd.log_gdp_std.max(), ls="--")
    rugged_seq = jnp.linspace(-0.1, 1.1, num=30)
    mu = Predictive(model, prior, return_sites=["mu"])(
        random.PRNGKey(7), rugged_std=rugged_seq
    )["mu"]
    for i in range(50):
        ax.plot(rugged_seq, mu[i], "C0", alpha=0.3)
axs[0].plot(rugged_seq, 1.3-.7*rugged_seq, "C1")
axs[0].set_title('a ~ Normal(1, 1)\nb ~ Normal(0, 1)')
axs[1].set_title('a ~ Normal(1, 0.1)\nb ~ Normal(0, 0.3)'){% endhighlight %}</details>

Kết quả được thể hiện ở bên trái [**HÌNH 8.3**](#f3). Đường nét đứt nằm ngang là tối đa và tối thiểu của giá trị log GDP quan sát. Những đường hồi quy có xu hướng cả dương và âm, như đáng lẽ nó vậy, nhưng đa số những đường này nằm trong vùng bất khả thi. Chỉ xem xét thang đo lường, những đường thẳng này nên đi qua gần điểm và độ gồ ghề là trung bình (0.125 ở trục hoành) và tỉ lệ log GDP là 1. Thay vào đó có nhiều đường mong đợi GDP trung bình nằm ngoài khoảng quan sát. Cho nên chúng ta cần prior $\alpha$ có độ lệch chuẩn hẹp hơn. $\alpha \sim \text{Normal}(0,0.1)$ sẽ đặt hầu hết tính phù hợp trong khoảng GDP quan sát được. Nhớ rằng: 95% của mật độ Gaussian là trong 2 độ lệch chuẩn. Cho nên prior Normal(0, 0.1) gán 95% tính phù hợp giữa 0.8 và 1.2. Nó vẫn còn mơ hồ, nhưng ít ra không phi lý.

Cùng lúc này, slope cũng rất biến thiên. Nó không phù hợp khi độ gồ ghề giải thích hầu như những biến thiên quan sát được trong log GDP. Một quan hệ mạnh không phù hợp, ví dụ, một đường thẳng đi từ độ gồ ghề tối thiểu và GDP cực ở một đầu đến độ gồ ghề tối đa và cực đối diện GDP ở đầu còn lại. tôi đã đánh dấu đường thẳng này bằng màu đỏ. Slope của đường này phải gần bằng 1.3 - 0.7 = 0.6, hiệu giữa tỉ lệ log GDP tối đa và tối thiểu quan sát được. Nhưng rất nhiều đường thẳng trong prior có slope mạnh hơn slope này. Dưới prior $\beta \sim \text{Normal}(0, 1)$, hơn một nửa slope sẽ có giá trị tuyệt đối lớn hơn 0.6.

<b>code 8.4</b>
```python
jnp.sum(jnp.abs(prior["b"]) > 0.6) / prior["b"].shape[0]
```
<samp>0.545</samp>

Hãy thay bằng $\beta \sim \text{Normal}(0, 0.3)$. Prior này có slope 0.6 là hai độ lệch chuẩn. Nó có một ít quá phù hợp, nhưng tốt hơn trước nhiều.

Với hai thay đổi này, mô hình bây giờ là:

<b>code 8.5</b>
```python
def model(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_1, losses = svi.run(random.PRNGKey(0), 1000)
```

Bạn có thể trích xuất prior và vẽ những đường thẳng suy ra bằng code như trước. Kết quả được thể hiện ở bên phải [**HÌNH 8.3**](#f3). Vài slope vẫn còn mạnh không phù hợp. Nhưng quan trọng, đây là tập prior tốt hơn. Nhìn vào posterior:

<b>code 8.6</b>
```python
post = m8_1.sample_posterior(random.PRNGKey(1), p8_1, (1000,))
print_summary({k: v for k, v in post.items() if k != "mu"}, 0.89, False)
```
<samp>       mean   std  median   5.5%  94.5%    n_eff  r_hat
    a  1.00  0.01    1.00   0.98   1.02   931.50   1.00
    b  0.00  0.06    0.00  -0.08   0.10  1111.63   1.00
sigma  0.14  0.01    0.14   0.13   0.15   949.29   1.00</samp>

Thực ra nhìn chung không có quan hệ giữa độ gồ ghề địa hình và log GDP. Tiếp theo chúng ta sẽ xem cách chia tách các lục địa.

<div class="alert alert-info">
<p><strong>Luyện tập cho lúc cần nó.</strong> Bài tập trong <a href="#f3"><strong>HÌNH 8.3</strong></a> thực ra không cần thiết trong ví dụ này, bởi vì có đủ data, và mô hình là đủ đơn giản, mà ngay cả prior tệ cũng bị mất tác dụng. Bạn thậm chí có thể sử dụng prior phẳng (đừng!), và nó vẫn tốt. Nhưng chúng ta luyện tập tốt thực hành này không phải vì nó luôn quan trọng. Mà là, chúng ta luyện tập làm việc này tốt để chúng ta chuẩn bị sẵn khi cần nó. Không ai nói rằng đeo đai bụng là sai lầm, chỉ bởi vì bạn không bị tai nạn.</p></div>

### 8.1.2 Thêm biến chỉ điểm là không đủ

Điều đầu tiên cần nhận ra là chỉ thêm biến chỉ điểm (indicator) cho quốc gia Châu Phi, `cont_africa` ở đây, sẽ không hiện ra hiện tượng đảo chiều của slope. Mặc dù bạn nên thử fit mô hình để tự chứng minh điều đó. Tôi sẽ đi qua phần này bằng bài tập so sánh mô hình đơn giản, để bạn bắt đầu có những ví dụ thực dụng của các khái niệm bạn đã tích luỹ từ những chương trước. Chú ý rằng so sánh mô hình ở đây không phải để chọn mô hình. Những kiến thức khoa học đã chọn ra mô hình liên quan. Thay vào đó nó là đo lường ảnh hưởng của sự khác nhau của mô hình khi quan tâm đến nguy cơ overfitting.

Để xây dụng mô hình cho phép quốc gia trong và ngoài Châu Phi để có những intercept khác nhau, chúng ta cần tuỳ chỉnh mô hình cho $\mu_i$ để trung bình được đặt điều kiện trên lục địa. Cách thuận tiện để làm điều này là chỉ cần thêm một số hạng khác vào mô hình tuyến tính:

$$\mu_i = \alpha +\beta(r_i -\bar{r}) + \gamma A_i$$

Trong đó $A-i$ là `cont_affrica`, một biến chỉ điểm 0/1. Nhưng đừng theo quy tắc này. Thực vậy, quy tắc này thông thường là một ý tưởng tệ. Tôi đã bỏ ra nhiều năm để nhận ra điều này, và tôi cố gắng cứu bạn khỏi nỗi kinh hoàng tôi đã gặp. Vấn đề là, tổng quát, chúng ta cần prior cho $\gamma$. Được rồi, chúng ta có thể làm prior. Nhưng những gì prior làm là nói cho mô hình biết $\mu_i$ cho một quốc gia trong Châu Phi là bất định hơn, trước khi thấy data, hơn $\mu_i$ ngoài Châu Phi. Và nó nghe hợp lý. Đây cũng giống như vấn đề chúng ta gặp ở Chương 4, khi tôi giới thiệu biến phân nhóm.

Đây là một giải pháp đơn giản: Quốc gia ở Châu Phi sẽ có một intercept và những quốc gia ngoài Châu Phi cũng như vậy. Lúc này $\mu_i$ sẽ có dạng:

$$\mu_i = \alpha_{\Tiny CID[i]} + \beta(r_i - \bar{i}) $$

Trong đó $CID$ là biến chỉ số, ID của lục địa. Nó nhận giá trị 0 cho quốc gia ở Châu Phi và 1 cho quốc gia khác. Điều này có nghĩa có 2 tham số, $\alpha_1$ à $\alpha_2$, mỗi một cho từng giá trị chỉ số độc nhất. Ký hiệu $CID[i]$ nghĩa là giá trị $CID$ ở hàng $i$. Tôi dùng ký hiệu ngoặc vuông cho biến chỉ số, bởi vì nó dễ hơn để đọc hơn thêm một dòng nằm dưới, $\alpha_{CID_i}$. Chúng ta có thể xây dựng biến chỉ số này như sau:

<b>code 8.7</b>
```python
# make variable to index Africa (0) or not (1)
dd["cid"] = jnp.where(dd.cont_africa.values == 1, 0, 1)
```

Qua tiếp cận này, thay vì dùng tiếp cận cũ bằng cách thêm một số hạng với biến chỉ điểm 0/1, không ràng buộc chúng ta nói rằng trung bình cho Châu Phi là ít tính bất định hơn trung bình của những lục địa khác. Chúng ta chỉ cần tái sử dụng prior như trước. Sau cùng, cho dù log GDP trung bình của Châu Phi là gì, nó chắc chắn nằm trong khoảng cộng-hoặc-trừ 0.2 của 1. Những hãy nhớ rằng nó là cùng một cấu trúc mô hình mà bạn có từ tiếp cận cũ. Theo cách này nó dễ dàng hơn trong việc gán prior hợp lý. Bạn có thể dễ dàng gán những prior khác cho lục địa khác, nếu bạn nghĩ rằng đó là điều nên làm.

Để định nghĩa mô hình bằng `SVI`, chúng ta thêm ngoặc vuông vào mô hình tuyến tính và prior:

<b>code 8.8</b>
```python
def model(cid, rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a[cid] + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_2 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_2,
    optim.Adam(0.1),
    Trace_ELBO(),
    cid=dd.cid.values,
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_2, losses = svi.run(random.PRNGKey(0), 1000)
```

Bây giờ hãy so sánh những mô hình này, sử dụng WAIC:

<b>code 8.9</b>
```python
post = m8_1.sample_posterior(random.PRNGKey(2), p8_1, (1000,))
logprob = log_likelihood(
    m8_1.model, post, rugged_std=dd.rugged_std.values, log_gdp_std=dd.log_gdp_std.values
)
az8_1 = az.from_dict({}, log_likelihood={k: v[None] for k, v in logprob.items()})
post = m8_2.sample_posterior(random.PRNGKey(2), p8_2, (1000,))
logprob = log_likelihood(
    m8_2.model,
    post,
    rugged_std=dd.rugged_std.values,
    cid=dd.cid.values,
    log_gdp_std=dd.log_gdp_std.values,
)
az8_2 = az.from_dict({}, log_likelihood={k: v[None] for k, v in logprob.items()})
az.compare({"m8_1": az8_1, "m8_2": az8_2}, ic="waic", scale="deviance")
```
<p><samp><table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>waic</th>
      <th>p_waic</th>
      <th>d_waic</th>
      <th>weight</th>
      <th>se</th>
      <th>dse</th>
      <th>warning</th>
      <th>waic_scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m8_2</th>
      <td>0</td>
      <td>-252.359592</td>
      <td>4.153888</td>
      <td>0.000000</td>
      <td>0.963947</td>
      <td>15.091342</td>
      <td>0.000000</td>
      <td>True</td>
      <td>deviance</td>
    </tr>
    <tr>
      <th>m8_1</th>
      <td>1</td>
      <td>-188.726754</td>
      <td>2.700302</td>
      <td>63.632839</td>
      <td>0.036053</td>
      <td>13.249023</td>
      <td>14.956883</td>
      <td>False</td>
      <td>deviance</td>
    </tr>
  </tbody>
</table></samp></p>

`m8_2` có tất cả những trọng số của mô hình. Và trong khi sai số chuẩn của hiệu số trong WAIC là 15, hiệu của chúng là 64. Cho nên biến lục địa có vẻ nhận được vài quan hệ quan trọng trong mẫu. Kết quả `print_summary` cho một gợi ý tốt. Chú ý `alpha` có hai giá trị. Thông thường một vector như vậy có vài trăm giá trị.

<b>code 8.10</b>
```python
post = m8_2.sample_posterior(random.PRNGKey(1), p8_2, (1000,))
print_summary({k: v for k, v in post.items() if k != "mu"}, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
 a[0]   0.88  0.02    0.88   0.86   0.90  1049.96   1.00
 a[1]   1.05  0.01    1.05   1.03   1.07   824.00   1.00
    b  -0.05  0.05   -0.05  -0.13   0.02   999.08   1.00
sigma   0.11  0.01    0.11   0.10   0.12   961.35   1.00</samp>

Tham số `a[0]` là intercept cho quốc gia ở Châu Phi. Nó nhỏ hơn `a[1]` một cách đáng tin cậy. Tương phản posterior giữa hai intercept này là:

<b>code 8.11</b>
```python
post = m8_2.sample_posterior(random.PRNGKey(1), p8_2, (1000,))
diff_a1_a2 = post["a"][:, 0] - post["a"][:, 1]
jnp.percentile(diff_a1_a2, q=(5.5, 94.5))
```
<samp>[-0.19981882, -0.13967244]</samp>

Hiệu số này nhỏ hơn zero đáng tin cậy. Hãy minh hoạ dự đoán posterior cho `m8_2`, để bạn thấy rằng, mặc dù khả năng dự đoán mạnh hơn so với `m8_1`, nó vẫn chưa nhận ra những slope khác nhau trong và ngoài Châu Phi. Để lấy mẫu từ posterior và tính trung bình và khoảng tin cậy của dự đoán cho quốc gia ở Châu Phi và không Châu Phi:

<b>code 8.12</b>
```python
rugged_seq = jnp.linspace(start=-0.1, stop=1.1, num=30)
# compute mu over samples, fixing cid=1
predictive = Predictive(m8_2.model, post, return_sites=["mu"])
mu_NotAfrica = predictive(random.PRNGKey(2), cid=1, rugged_std=rugged_seq)["mu"]
# compute mu over samples, fixing cid=0
mu_Africa = predictive(random.PRNGKey(2), cid=0, rugged_std=rugged_seq)["mu"]
# summarize to means and intervals
mu_NotAfrica_mu = jnp.mean(mu_NotAfrica, 0)
mu_NotAfrica_ci = jnp.percentile(mu_NotAfrica, q=(1.5, 98.5), axis=0)
mu_Africa_mu = jnp.mean(mu_Africa, 0)
mu_Africa_ci = jnp.percentile(mu_Africa, q=(1.5, 98.5), axis=0)
```

<a name="f4"></a>![](/assets/images/fig 8-4.svg)
<details class="fig"><summary>Hình 8.4: Thêm một biến chỉ điểm cho quốc gia ở Châu Phi không có ảnh hưởng lên slope. Quốc gia ở Châu Phi màu xanh, quốc gia không Châu Phi màu đỏ. Trung bình hồi quy cho mỗi nhóm quốc gia được hiện theo màu tương ứng, với khoảng tin cậy 97%.</summary>
{% highlight python %}cond = dd['cont_africa']==0
plt.scatter(dd['rugged_std'][~cond], dd['log_gdp_std'][~cond], color="C0")
plt.scatter(dd['rugged_std'][cond], dd['log_gdp_std'][cond], color="C1")
plt.plot(rugged_seq,mu_Africa_mu,'C0')
plt.plot(rugged_seq,mu_NotAfrica_mu,'C1')
plt.fill_between(rugged_seq, *mu_Africa_ci, color="C0", alpha=0.3)
plt.fill_between(rugged_seq, *mu_NotAfrica_ci, color="C1", alpha=0.3)
plt.gca().set(title='m8_4',xlabel="độ gồ ghề (chuẩn hoá)",
              ylabel="log GDP (tỉ lệ với trung bình)")
plt.annotate("Châu Phi", (0.8, 0.8))
plt.annotate("không Châu Phi", (0.8, 0.98)){% endhighlight %}</details>

Tôi thể hiện những dự đoán posterior (dự đoán ngược) ở [**HÌNH 8.4**](#f4). Quốc gia ở Châu Phi là màu xanh, quốc gia ngoài Châu Phi là màu đỏ. Bạn có được ở đây là một quan hệ yếu giữa kinh tế và độ gồ ghề. Quốc gia ở Châu Phi nhìn chung có phát triển kinh tế thấp hơn, và nên đường hồi quy màu xanh là nằm dưới, nhưng song song với, đường màu đỏ. Tất cả bao gồm biến giả cho quốc gia ở Châu Phi đã làm là cho phép mô hình dự đoán trung bình thấp hơn cho quốc gia ở Châu Phi. Nó không thể làm gì cho slope của đường thẳng. Sự thật WAIC nói chúng ta biết rằng mô hình với biến giả là tốt hơn rất nhiều với mô hình chỉ nói lên rằng quốc gia ở Châu Phi có trung bình GDP thấp hơn.

<div class="alert alert-info">
<p><strong>Tại sao 97%?</strong> Trong code trên cũng như trong <a href="#f4"><strong>HÌNH 8.4</strong></a>, tôi sử dụng khoảng 97% của trung bình mong đợi. Đây là một khoảng bách phân không tiêu chuẩn. Tại sao lại dùng 97%? Trong sách này, tôi dùng phần trăm không tiêu chuẩn để luôn luôn nhắc nhở người đọc rằng khoảng tiện lợi như 95% và 5% là ngẫu nhiên. Hơn nữa, biên giới này là vô nghĩa. Có một sự thay đổi liên tục trong xác suất khi chúng ta rời xa giá trị mong đợi. Cho nên một mặt của biên giới là gần bằng xác suất của bên còn lại. Và, 97 là số nguyên tố. Nó không có nghĩa là lựa chọn tốt hơn những con số khác ở đây, nhưng nó không ngu ngốc hơn khi sử dụng bội số của 5, chỉ bởi vì chúng ta có 5 ngón ở mỗi bàn tay. Hãy chống lại sự cai trị của Tetrapoda.</p></div>

### 8.1.3 Việc thêm sự tương tác là đúng

Làm sao để phục hồi sự thay đổi slope bạn đã nhìn thấy ở đầu phần này? Bạn cần hiệu ứng tương tác rõ ràng. Điều này nghĩa là chúng ta phải làm cho slope được đặt điều kiện trên lục địa. Định nghĩa của $\mu_i$ trong mô hình bạn đã minh hoạ, dưới dạng toán học là:

$$ \mu_i = \alpha_{CID[i]} + \beta(r_i - \bar{r})$$

Và bây giờ chúng ta sẽ nhân đôi chỉ số của chúng ta để làm cho slope cũng được đặt điều kiện:

$$ \mu_i = \alpha_{CID[i]} + \beta_{CID[i]}(r_i - \bar{r})$$

Và lần nữa, đây là một tiếp cận thuận tiện để xác định tương tác có sử dụng biến chỉ điểm và một biến tương tác mới. Nó trông giống như vậy:

$$ \mu_i = \alpha_{CID[i]} + (\beta+\gamma A_i)(r_i - \bar{r})$$

Trong khi $A_i$ là biến chỉ điểm 0/1 cho quốc gia ở Châu Phi. Nó là tương đương với cách tiếp cận chỉ số, nhưng khó hơn để đưa ra prior hợp lý. Bất kỳ prior chúng ta đặt lên $\gamma$ sẽ làm slope trong Châu Phi có tính bất định cao hơn slope ngoài Châi Phi. Và lần nữa nó vô lý. Nhưng trong cách tiếp cận chỉ số, chúng ta có thể dễ dàng gán cùng một prior cho slope, cho dù lục địa nào.

Để ước lượng posterior cho mô hình mới, chúng ta vẫn sử dụng `SVI` như trước. Đây là code bao gồm tương giữa độ gồ ghề và ở Châu Phi:

<b>code 8.13</b>
```python
def model(cid, rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1).expand([2]))
    b = numpyro.sample("b", dist.Normal(0, 0.3).expand([2]))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a[cid] + b[cid] * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_3 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_3,
    optim.Adam(0.1),
    Trace_ELBO(),
    cid=dd.cid.values,
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_3, losses = svi.run(random.PRNGKey(0), 1000)
```

Hãy kiểm tra phân phối posterior biên:

<b>code 8.14</b>
```python
post = m8_3.sample_posterior(random.PRNGKey(1), p8_3, (1000,))
print_summary({k: v for k, v in post.items() if k != "mu"}, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
 a[0]   0.89  0.02    0.89   0.86   0.91  1009.20   1.00
 a[1]   1.05  0.01    1.05   1.04   1.07   755.33   1.00
 b[0]   0.13  0.07    0.13   0.01   0.24  1045.06   1.00
 b[1]  -0.15  0.06   -0.14  -0.23  -0.05  1003.36   1.00
sigma   0.11  0.01    0.11   0.10   0.12   810.01   1.00</samp>

Slope đã quay ngược đúng trong Châu Phi, 0.13 thay vì -0.14.

Việc cho phép slope thay đổi đã cải thiện dự đoán mong đợi như thế nào? Hãy dùng PSIS để so sánh mô hình mới này với hai mô hình trước. Bạn cũng có thể dùng WAIC. Nó sẽ cho kết quả giống nhau. Nhưng nó không cho cảnh báo Pareto $k$ ngọt ngào.

<b>code 8.15</b>
```python
post = m8_1.sample_posterior(random.PRNGKey(2), p8_1, (1000,))
logprob = log_likelihood(
    m8_1.model, post, rugged_std=dd.rugged_std.values, log_gdp_std=dd.log_gdp_std.values
)
az8_1 = az.from_dict({}, log_likelihood={k: v[None] for k, v in logprob.items()})
post = m8_2.sample_posterior(random.PRNGKey(2), p8_2, (1000,))
logprob = log_likelihood(
    m8_2.model,
    post,
    rugged_std=dd.rugged_std.values,
    cid=dd.cid.values,
    log_gdp_std=dd.log_gdp_std.values,
)
az8_3 = az.from_dict({}, log_likelihood={k: v[None] for k, v in logprob.items()})
post = m8_3.sample_posterior(random.PRNGKey(2), p8_3, (1000,))
logprob = log_likelihood(
    m8_3.model,
    post,
    rugged_std=dd.rugged_std.values,
    cid=dd.cid.values,
    log_gdp_std=dd.log_gdp_std.values,
)
az8_3 = az.from_dict({}, log_likelihood={k: v[None] for k, v in logprob.items()})
az.compare({"m8_1": az8_1, "m8_2": az8_2, "m8_3": az8_3}, ic="waic", scale="deviance")
```
<p><samp><table border="1" class="dataframe">
<thead><tr style="text-align: right;">
<th></th>
      <th>rank</th>
      <th>waic</th>
      <th>p_waic</th>
      <th>d_waic</th>
      <th>weight</th>
      <th>se</th>
      <th>dse</th>
      <th>warning</th>
      <th>waic_scale</th>
    </tr></thead>
<tbody>
<tr>
<th>m8_3</th>
      <td>0</td>
      <td>-259.176</td>
      <td>5.10348</td>
      <td>0</td>
      <td>0.824888</td>
      <td>13.4328</td>
      <td>0</td>
      <td>True</td>
      <td>deviance</td>
    </tr>
<tr>
<th>m8_2</th>
      <td>1</td>
      <td>-252.36</td>
      <td>4.15389</td>
      <td>6.81647</td>
      <td>0.175111</td>
      <td>14.6901</td>
      <td>6.67691</td>
      <td>True</td>
      <td>deviance</td>
    </tr>
<tr>
<th>m8_1</th>
      <td>2</td>
      <td>-188.818</td>
      <td>2.65329</td>
      <td>70.3582</td>
      <td>4.90447e-08</td>
      <td>14.6588</td>
      <td>15.3423</td>
      <td>False</td>
      <td>deviance</td>
    </tr>
</tbody>
</table></samp></p>

Gia đình mô hình `m8_3` có hơn 95% trọng số. Đó là một ủng hộ mạnh cho việc bao gồm hiệu ứng tương tác, nếu dự đoán là mục đích của chúng ta. Nhưng giá trị trọng số cho `m8_2` đề nghị rằng trung bình posterior cho slope ở `m8_3` có một ít overfit. Và sai số chuẩn của hiệu số trong PSIS giữa hai mô hình trên là hầu như bằng nhau với bản thân hiệu số. Nếu bạn vẽ PSIS Pareto $k$ cho `m8_3`, bạn sẽ thấy những quốc gia có ảnh hưởng.

<b>code 8.16</b>
```python
plt.plot(az.loo(az8_3, pointwise=True).pareto_k.data)
```

Bạn sẽ khám phá vấn đề này trong phần thực hành cuối chương. Nó là một tình huống tốt cho hồi quy robust, như hồi quy Student_t chúng ta đã làm ở Chương 7.

Nên nhớ rằng việc so sánh là không phải chỉ dẫn tin cậy cho suy luận nhân quả. Chúng chỉ gợi ý những đặc trưng quan trọng cho dự đoán. Hiệu ứng nhân quả thực sự có thể không quan trọng cho dự đoán chung với bất kỳ mẫu nào. Dự đoán và suy luận là hai câu hỏi khác nhau. Cho dù thế nào, overfitting luôn luôn xảy ra. Cho nên lường trước và đo lường nó cũng quan trọng cho suy luận.

### 8.1.4 Biểu đồ của sự tương tác

Minh hoạ cho mô hình này không cần mánh gì mới. Mục tiêu là làm hai biểu đồ. Trong biểu đồ đầu tiên, chúng ta sẽ thể hiện quốc gia Châu Phi và thêm một lớp đường trung bình và khoảng tin cậy 97% của hồi quy. Trong biểu đồ thứ hai, chúng ta sẽ thể hiện quốc gia ngoài Châu Phi với cách làm tương tự.

<b>code 8.17</b>
```python
post = m8_3.sample_posterior(random.PRNGKey(1), p8_3, (1000,))
rugged_seq = jnp.linspace(start=-0.1, stop=1.1, num=30)
predictive = Predictive(m8_3.model, post, return_sites=["mu"])
fig, axs= plt.subplots(1,2,figsize=(10,5))
for i,ax in enumerate(axs):
    mu = predictive(random.PRNGKey(2), cid=i, rugged_std=rugged_seq)["mu"]
    mu_mean = jnp.mean(mu, axis=0)
    mu_ci = jnp.percentile(mu, jnp.array([1.5, 98.5]), axis=0)
    cond =dd['cid']==i
    ax.scatter(dd[cond]['rugged_std'], dd[cond]['log_gdp_std'], color=f"C{i}")
    ax.plot(rugged_seq, mu_mean, color=f'C{i}')
    ax.fill_between(rugged_seq, *mu_ci, color=f'C{i}', alpha=0.3)
    ax.set(xlabel="độ gồ ghề (chuẩn hoá)", ylabel="log GDP (tỉ lệ với trung bình)")
    for j in range(0,50,10):
        n = dd[cond].iloc[j]
```

<a name="f5"></a>![](/assets/images/fig 8-5.svg)
<details class="fig"><summary>Hình 8.5: Dự đoán posterior cho mô hình gồ ghề địa hình, bao gồm tương tác giữa Châu Phi và độ gồ ghề. Vùng tô màu là khoảng posterior 97% của trung bình.</summary>
{% highlight python %}post = m8_3.sample_posterior(random.PRNGKey(1), p8_3, (1000,))
rugged_seq = jnp.linspace(start=-0.1, stop=1.1, num=30)
predictive = Predictive(m8_3.model, post, return_sites=["mu"])
fig, axs= plt.subplots(1,2,figsize=(10,5))
for i,ax in enumerate(axs):
    mu = predictive(random.PRNGKey(2), cid=i, rugged_std=rugged_seq)["mu"]
    mu_mean = jnp.mean(mu, axis=0)
    mu_ci = jnp.percentile(mu, jnp.array([1.5, 98.5]), axis=0)
    cond =dd['cid']==i
    ax.scatter(dd[cond]['rugged_std'], dd[cond]['log_gdp_std'], color=f"C{i}")
    ax.plot(rugged_seq, mu_mean, color=f'C{i}')
    ax.fill_between(rugged_seq, *mu_ci, color=f'C{i}', alpha=0.3)
    ax.set(xlabel="độ gồ ghề (chuẩn hoá)", ylabel="log GDP (tỉ lệ với trung bình)")
    for j in range(0,50,10):
        n = dd[cond].iloc[j]
        ax.annotate(n['country'],(n['rugged_std'], n['log_gdp_std']))
axs[0].set(title="Quốc gia Châu Phi")
axs[1].set(title="Quốc gia không Châu Phi"){% endhighlight %}</details>

Và kết quả được thể hiện ở [**HÌNH 8.5**](#f5). Cuối cùng, sự đảo chiều slope dã xảy ra trong và ngoài Châu Phi. Và bởi vì chúng ta đạt được điều này chỉ trong một mô hình duy nhất, chúng ta có thể lượng giá ý nghĩa của sự đảo chiều này bằng thống kê.

## <center>8.2 Tính đối xứng của sự tương tác</center><a name="a2"></a>

Con lừa của Buridan là một câu đố triết học trong đó một con lừa luôn luôn đi về phần thức ăn gần nhất sẽ chết đói khi gặp trường hợp đứng giữa hai phần thức ăn cách xa như nhau. Vấn đề cơ bản là tính đối xứng: Con lừa quyết định giữa hai lựa chọn giống hệt nhau như thế nào? Giống như nhiều vấn đề khác, bạn không nên đặt nặng vấn đề này. Dĩ nhiên, con lừa sẽ không chết đói. Nhưng suy nghĩ về tại sao tính đối xứng bị phá vỡ có thể mang lợi ích.

Sự tương tác giống như con lừa của Buridan. Giống như hai phần thức ăn giống nhau, một mô hình tương tác đơn giản gồm hai cách diễn giải đối xứng nhau. Khi thiếu vắng những thông tin khác, ngoài mô hình, thì không có cơ sở logic nào để ưu ái một diễn giải này hơn diễn giải kia. Xem xét ví dụ GDP và mức độ gồ ghề địa hình. Sự tương tác ở đây có hai cách diễn giải tương tự nhau.

1. Quan hệ giữa độ gồ ghề và log GDP là phụ thuộc vào quốc gia đó có trong Châu Phi hay không.
2. Quan hệ giữa Châu Phi và log GDP là phụ thuộc vào độ gồ ghề.

Trong khi hai khả năng này nghe có vẻ khác nhau với nhiều người, golem của bạn nghĩ chúng là như nhau. Trong phần này, chúng ta sẽ khám phá hiện tượng này, bằng toán học. Sau đó chúng ta sẽ minh hoạ ví dụ độ gồ ghề và GDP lần nữa, nhưng với các diễn giải ngược lại - quan hệ giữa Châu Phi và GDP phụ thuộc vào độ gồ ghề.

Xem xét mô hình cho $\mu_i$ lần nữa:

$$\mu_i = \alpha_{CID[i]} + \beta_{CID[i]}(r_i - \bar{r})$$

Diễn này trước nói rằng slope được đặt điều kiện trên lục địa. Nhưng cũng có thể nói rằng intercept được đặt điều kiện trên đồ gồ ghề. Sẽ dễ nhìn ra hơn nếu chúng ta viết lại biểu thức trên theo cách khác:

$$\mu_i = \underbrace{(2-CID_i)(\alpha_1 + \beta_1(r_i -\bar{r}))}_{CID[i]=0} + \underbrace{(CID_i -1)(\alpha_2 +\beta2(r_i -\bar{r}))}_{CID[i]=1}$$

Nó có vẻ lạ, nhưng là cùng một mô hình. Khi $CID_i=0$, chỉ có số hạng đầu tiên, tham số Châu Phi, là ở lại. Số hạng thứ hai biến mất thành zero. Khi $CID_i=1$, số hạng đầu tiên biến mất thành zero, chỉ còn lại số hạng thứ hai. Bây giờ nếu chúng ta tưởng tượng thau đổi một quốc gia thành ở Châu Phi, để biết được nó ảnh hưởng gì đến dự đoán, chúng ta phải biết độ gồ ghề (trừ phi chúng ta đang ở ngay độ gồ ghề trung bình, $\bar{r}$).

Sẽ có ích nếu minh hoạ ra cách diễn đạt ngược này: *Quan hệ giữa ở Châu Phi và log GDP phụ thuộc vào độ gồ ghề địa hình*. Chúng ta sẽ tính sự khác nhau giữa quốc gia ở Châu Phi và ngoài Châu Phi, với độ gồ ghề hằng định. Để làm việc này, bạn có thể hạy `Predictive` hai lần và lấy kết quả đầu tiên trừ đi kết quả thứ hai:

<b>code 8.18</b>
```python
rugged_seq = jnp.linspace(start=-0.2, stop=1.2, num=30)
post = m8_3.sample_posterior(random.PRNGKey(1), p8_3, (1000,))
predictive = Predictive(m8_3.model, post, return_sites=["mu"])
muA = predictive(random.PRNGKey(2), cid=0, rugged_std=rugged_seq)["mu"]
muN = predictive(random.PRNGKey(2), cid=1, rugged_std=rugged_seq)["mu"]
delta = muA - muN
```

Bạn có thể tóm tắt lại và vẽ biểu đồ hiệu của log GDP mong đợi nằm trong `delta`.

<a name="f6"></a>![](/assets/images/fig 8-6.svg)
<details class="fig"><summary>Hình 8.6: Mặc khác của sự tương tác giữa đồ gồ ghề và lục địa. Trục tung là hiệu giữa tỉ lệ log GDP mong đợi cho một quốc gia ở Châu Phi và một quốc gia ngoài Châu Phi. Ở độ gồ ghề thấp, chúng ta mong đợi việc "di chuyển" một quốc gia sang Châu Phi sẽ làm tổn hại đến kinh tế của nó. Nhưng với đồ gồ ghề cao, điều ngược lại là đúng. Quan hệ giữa lục địa và kinh tế phụ thuộc vào độ gồ ghề, cũng giống như quan hệ giữa đồ gồ ghề và kinh tế phụ thuộc vào lục địa.</summary>
{% highlight python %}plt.plot(rugged_seq, jnp.mean(delta,axis=0))
plt.fill_between(rugged_seq, *jnp.quantile(delta, q=jnp.array([0.025, 0.975]), axis=0), alpha=0.3)
plt.hlines(0,-0.2,1.2, lw=3, ls='dashed')
plt.annotate('Châu Phi GDP cao', (-0.2, 0.02))
plt.annotate('Châu Phi GDP thấp', (-0.2, -0.06))
plt.xlabel('độ gồ ghề')
plt.ylabel('hiệu log GDP mong đợi'){% endhighlight %}</details>

Kết quả được thể hiện ở [**HÌNH 8.6**]. Biểu đồ này là *phản thực*. Không có data thô ở đây. Thay vào đó chúng ta đang nhìn qua con mắt của mô hình và đang tưởng tượng so sánh giữa những quốc gia giống hệt nhau nằm trong và ngoài Châu phi, như thể chúng ta có thể tự do điều kiện lục địa và cũng như độ gồ ghề. Dưới đường nét đứt ngang, những quốc gia ở Châu Phi có GDP mong đợi thấp hơn. Đó là trường hợp cho phần lớn giá trị độ gồ ghề. Nhưng ở những giá trị độ gồ ghề cao, một quốc gia có thể tốt hơn khi ở trong Châu Phi hơn là ở ngoài. Thực vậy rất khó để tìm ra sự khác nhau đáng tin cậy giữa trong và ngoài Châu Phi, ở giá trị độ gồ ghề cao. Nó là một quốc gia có nền kinh tế tốt chỉ khi ở trong Châu Phi.

Khía cạnh này của GDP và độ gồ ghề là hoàn toàn hằng định với khía cạnh trước. Nó đồng thời đúng trong data này (và mô hình) mà (1) ảnh hưởng của đồ gồ ghề phụ thuộc lục địa và (2) ảnh hưởng của lục địa phụ thuộc vào độ gồ ghề. Thực vậy, bạn đã nhận được một thứ gì đó bằng cách nhìn vào data qua khía cạnh đối xứng này. Bằng cách kiểm tra góc nhìn đầu tiên của sự tương tác, nó không rõ ràng là quốc gia Châu Phi trên trung bình là luôn luôn tệ đi. Chỉ khi giá trị `rugged` rất cao thì lúc đó quốc gia trong và ngoài Châu Phi có log GDP mong đợi như nhau. Cách minh hoạ tương tác thứ hai làm điều này rõ hơn.

Tương tác đơn giản là đối xứng, nhưng như lựa chọn của con lừa của Buridan. Trong mô hình, không có cơ sở nào ưu ái diễn giải nào hơn diễn giải còn lại, bởi vì sự thật là chúng cùng là một diễn giải. Nhưng khi chúng ta lý giải nhân quả về mô hình, ý thức của chúng ta thường ưu ái một diễn giải nào đó hơn, bởi vì nó dễ hơn khi tưởng tượng điều khiển một biến dự đoán hơn những biến khác. Trong trường hợp này, khó tưởng tượng điều khiển lục địa mà quốc gia ở trong. Nhưng dễ tưởng tượng điều khiển độ gồ ghề địa hình, như làm phẳng các đồi núi hoặc xây dựng những đường hầm qua núi.<sup><a name="r141" href="#141">141</a></sup> Nếu thực ra cách giải thích cho mối quan hệ dương bất thường của Châu Phi với mức độ gồ ghề là do nguyên nhân lịch sử, không phải đồng thời do địa hình, thì những đường hầm có thể cải thiện kinh tế hiện tại. Đồng thời, lục địa không thực sự là một nguyên nhân của hoạt động kinh tế. Thực ra có nhiều yếu tố lịch sử và chính trị liên quan đến lục địa, và chúng ta sử dụng biến lục địa là một đại diện trung gian cho những yếu tố đó. Việc điều khiển những yếu tố khác đó mới quan trọng.

## <center>8.3 Sự tương tác liên tục</center><a name="a3"></a>

Tôi muốn thuyết phục người đọc rằng hiệu ứng tương tác rất khó để diễn giải. Diễn giải chúng là hầu như bất khả thi, chỉ bằng trung bình và độ lệch chuẩn posterior. Một khi sự tương tác tồn tại, nhiều tham số cùng lúc ở trong cuộc chơi. Nó đủ khó với sự tương tác đơn giản, phân nhóm trong ví dụ độ gồ ghề địa hình. Một khi chúng ta bắt đầu mô hình hoá sự tương tác giữa các biến liên tục, nó khó hơn rất nhiều. Chỉ có một thứ để một slope được đặt điều kiện trên một *nhóm*. Trong bối cảnh này, mô hình thu gọn thành ước lượng slope khác nhau cho mỗi nhóm. Nhưng sẽ khó hơn nhiều để hiểu slope biến thiên trong kiểu liên tục với một biến liên tục. Diễn giải khó hơn nhiều trong trường hợp này, mặc dù công thức toán học của mô hình là như nhau.

Để làm rõ ràng việc xây dựng và diễn giải **SỰ TƯƠNG TÁC LIÊN TỤC** giữa hai hay nhiều biến dự đoán liên tục, trong phần này tôi phát triển một ví dụ hồi quy đơn giản và cho bạn thấy cách minh hoạ sự tương tác hai chiều giữa hai biến liên tục. Phương pháp tôi trình bày cho việc vẽ biểu đồ cho sự tương tác này là biểu đồ *triptych*, một bảng có hình vẽ bổ trợ cho nhau tạo thành toàn bộ bức tranh cho kết quả hồi quy. Không có gì đặc biệt khi có ba hình - trong những trường hợp khác bạn có thể cần nhiều hoặc ít hơn. Thay vào đó, chức năng chính nằm ở việc tạo nhiều hình cho phép chúng ta nhìn thấy sự tương tác thay đổi slope như thế nào, khi có sự thay đổi của một biến được chọn.

### 8.3.1 Đoá hoa mùa đông

Data trong ví dụ này là kích cỡ của những bông hoa tulip trồng trong nhà kính, với điều kiện đất và ánh sáng khác nhau.<sup><a name="r142" href="#142">142</a></sup> Tải data về:

<b>code 8.19</b>
```python
url = r'https://github.com/rmcelreath/rethinking/blob/master/data/tulips.csv'
d = pd.read_csv(url+"?raw=true",sep=';')
d.info()
d.head()
```
<samp>&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 27 entries, 0 to 26
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   bed     27 non-null     object 
 1   water   27 non-null     int64  
 2   shade   27 non-null     int64  
 3   blooms  27 non-null     float64
dtypes: float64(1), int64(2), object(1)
memory usage: 992.0+ bytes</samp>
<p><samp><table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bed</th>
      <th>water</th>
      <th>shade</th>
      <th>blooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>1</td>
      <td>3</td>
      <td>111.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>2</td>
      <td>1</td>
      <td>183.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>2</td>
      <td>2</td>
      <td>59.16</td>
    </tr>
  </tbody>
</table></samp></p>

Cột `blooms` chứa kết cục của chúng ta - thứ chúng ta muốn dự đoán. Cột `water` và `shade` sẽ là biến dự đoán. `water` chỉ điểm cho một trong ba mức độ ẩm của đất, từ thấp (1) đến cao (3). `shade` chỉ điểm cho một trong ba mức độ phơi nhiễm ánh sáng, từ cao (1) đến thấp (3). Cột cuối cùng, `bed`, chỉ điểm cho một cụm cây nằm chung một khu của nhà kính.

Bởi vì ánh sáng và nước giúp cây tăng trưởng và nở hoa, nó hợp lý khi hiệu ứng độc lập của mỗi yếu tố sẽ giúp tạo ra hoa to hơn. Nhưng chúng ta cũng quan tâm đến sự tương tác giữa hai biến này. Khi thiết hụt ánh sáng, lấy ví dụ, thì khó thấy hơn nước giúp cây như thế nào - quang hợp phụ thuộc vào cả ánh sáng và nước. Tương tự, khi thiết hụt nước, ánh sáng mặt trời không giúp cho cây nhiều. Một cách để mô hình sự phụ thuộc lẫn nhau này là dùng hiệu ứng tương tác. Trong khi không có một mô hình cơ học tốt cho sự tương tác này, mô hình mà sử dụng lý thuyết về sinh lý thực vật để đặt giải thuyết quan hệ chức năng giữa ánh sáng và nước, thì một mô hình tuyến tính đơn giản có tương tác hai chiều là một khởi đầu tốt. Nhưng suy cho cùng nó không gần với thứ tốt nhất mà chúng ta có thể làm.

### 8.3.2 Những mô hình

Tôi sẽ tập trung vào chỉ hai mô hình: (1) mô hình có `water` và `shade` nhưng không có sự tương tác và (2) mô hình có chứa sự tương tác giữa `water` và `shade`. Bạn cũng có thể kiểm tra mô hình chứa chỉ một trong những biết này, `water` hoặc `shade`, và tôi khuyến khích người đọc thử nó vào cuối chương và đảm bảo rằng bạn hiểu cách lắp ráp mô hình.

Tình huống nhân quả đơn giản là nước ($W$) và bóng ($S$) đều ảnh hưởng đến hoa ($B$):  $W \to B \gets S$. Như trước, DAG này không nói cho chúng ta biết chức căng của $W$ và $S$ ảnh hưởng kết hợp lên $B$, $B=f(W,S)$. Theo nguyên tắc, mọi sự kết hợp độc nhất của $W$ và $S$ đều có thể có trung bình $B$ khác. Cách thông thường là làm gì đó đơn giản hơn nhiều. Chúng ta sẽ bắt đầu đơn giản.

Mô hình đầu tiên, không chứa sự tương tác nào cả (chỉ có "hiệu ứng chính"), bắt đầu bằng cách này:

$$\begin{aligned}
\beta_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta_W (W_i -\bar{W}) + \beta_S (S_i - \bar{S})\\
\end{aligned}$$

Trong đó \beta_i là giá trị của `blooms` ở hàng $i$, W_i là giá trị của `water`, và $S_i$ là giá trị của `shade`. Ký hiệu $\bar{W}$ và $\bar{S}$ lần lượt là trung bình của nước và bóng. Tất cả những thứ này, là hồi quy tuyến tính với hai biến đự đoán, mỗi một được canh giữa bằng cách trừ đi trung bình của chúng.

Để cho ước lượng dễ hơn, hãy canh giữa $W$ và $S$ và chỉnh thang đo $B$ bằng giá trị cực đại của nó:

<b>code 8.20</b>
```python
d["blooms_std"] = d.blooms / d.blooms.max()
d["water_cent"] = d.water - d.water.mean()
d["shade_cent"] = d.shade - d.shade.mean()
```

Bây giờ `blooms_std` có khoảng từ 0 đến 1, và cả `water_cent` và `shade_cent` đều có khoảng từ -1 đến 1. Tôi đã chỉnh thang đo `blooms` bằng giá trị cực đại quan sát được của nó, vì ba lý do. Đầu tiên, giá trị lớn ở thang đo thô sẽ làm cho việc tối ưu hoá khó khăn hơn. Thứ hai, nó sẽ dễ hơn khi gán prior hợp lý. Thứ ba, chúng ta không muốn chuẩn hoá `blooms`, vì giá trị zero là một biên có ý nghĩa mà chúng ta muốn giữ lại.

Khi thay đổi thang đo của biến số, một mục tiêu tốt là tạo ra những điểm khu trú mà bạn có thông tin prior, prior trước khi thấy data thực thụ. Bằng cách này, chúng ta có thể gán những prior không quá điên rồ. Và bằng cách suy nghĩ về prior, chúng ta có thể nhận ra mô hình không hợp lý. Nhưng điều này chỉ có thể nếu chúng ta nghĩ về quan hệ giữa đo lường và tham số. Bài tập về thay đổi thang đo và gán prior sẽ có ích. Mặc dù khi data đủ lớn đến nỗi lựa chọn prior không ảnh hưởng nhiều, bài tập suy nghĩ này vẫn có lợi.

Có ba tham số (ngoại trừ $\alpha$) trong mô hình này, cho nên chúng ta cần ba prior. Trước tiên hãy đoán mò:

$$\begin{aligned}
\alpha &\sim \text{Normal}(0.5,1)\\
\beta_W &\sim \text{Normal}(0,1)\\
\beta_S &\sim \text{Normal}(0,1)\\
\end{aligned}$$

Canh giữa prior của $\alpha$ ở 0.5 suy ra rằng, khi cả nước và bóng ở giá trị trung bình của chúng, mô hình dự đoán rằng hoa sẽ ở giữa đến cực đại quan sát được. Hai slope được canh giữa ở zero, gợi ý rằng không có thông tin prior nào về chiều hướng. Điều này rõ ràng ít thông tin hơn những gì chúng ta có - kiến thức trồng cây cơ bản cho ta biết rằng nước nên có slope dường và bóng có slope âm. Những những prior này cho phép chúng ta nhìn thấy xu hướng nào mà mẫu thể hiện ra, trong khi vẫn bị ràng buộc bởi những giá trị hợp lý. Trong thực hành cuối chương, tôi sẽ yêu cầu bạn sử dụng kiến thức trồng cây của bạn.

Biên giới prior của tham số đến từ độ lệch chuẩn của prior, tất cả đều được đặt bằng 1 ở đây. Chúng chắc chắn là quá rộng. Intercept $\alpha$ phải lớn hơn zero và nhỏ hơn 1, ví dụ. Nhưng prior này gán hầu hết các xác suất ngoài khoảng này:

<b>code 8.21</b>
```python
a = dist.Normal(0.5, 1).sample(random.PRNGKey(0), (int(1e4),))
jnp.sum((a < 0) | (a > 1)) / a.shape[0]
```
<samp>0.6182</samp>

Nếu nó là 0.5 đơn vị từ trung bình đến zero, thì độ lệch chuẩn 0.25 sẽ đặt 5% mật độ ngoài khoảng phù hợp. Hãy xem:

<b>code 8.22</b>
```python
a = dist.Normal(0.5, 0.25).sample(random.PRNGKey(0), (int(1e4),))
jnp.sum((a < 0) | (a > 1)) / a.shape[0]
```
<samp>0.0471</samp>

Khá hơn rồi. Còn slope thì sao? Hiệu ứng mạnh của nước và bóng sẽ trông ra sao? Theo lý thuyết thì những slope có thể lớn như thế nào? Khoảng của cả nước và bóng là 2 - từ -1 đến 2 đơn vị. Để đưa chúng ta từ điểm cực tiểu là zero theo lý thuyết ở một đầu đến điểm cực đại quan sát được là 1 - một khoảng dài 1 đơn vị - ở đầu còn lại sẽ cần slope 0.5 từ cả hai biến - $0.5 \times 2 =1$. Cho nên nếu chúng ta gán độ lệch chuẩn 0.25 lên hai biến đó, thì 95% của slope prior là từ -0.5 đến 0.5, để cả hai biến có thể theo nguyên tắc đáp ứng cho toàn khoảng, nhưng nó sẽ rất ít xảy ra. Cần nhớ rằng, mục tiêu ở đây là gán prior chứa thông tin yếu để hạn chế overfitting - những hiệu ứng quá lớn nên được gán xác suất prior thấp - và cũng để ép buộc chúng ta nghĩ về ý nghĩa của mô hình.

Tất cả những lý luận trên, ở dạng code:

<b>code 8.23</b>
```python
def model(water_cent, shade_cent, blooms_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 0.25))
    bw = numpyro.sample("bw", dist.Normal(0, 0.25))
    bs = numpyro.sample("bs", dist.Normal(0, 0.25))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bw * water_cent + bs * shade_cent)
    numpyro.sample("blooms_std", dist.Normal(mu, sigma), obs=blooms_std)
m8_4 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_4,
    optim.Adam(1),
    Trace_ELBO(),
    shade_cent=d.shade_cent.values,
    water_cent=d.water_cent.values,
    blooms_std=d.blooms_std.values,
)
p8_4, losses = svi.run(random.PRNGKey(0), 1000)
```

Ở thời điểm này thì việc mô phỏng các đường thẳng từ prior là một ý tưởng tốt. Nhưng trước khi thực hiện, hãy định nghĩa mô hình tương tác luôn. Sau đó chúng ta có thể nói về làm sao vẽ dự đoán từ sự tương tác và xem cả dự đoán prior và posterior cùng lúc.

Để xây dựng tương tác giữa nước và bóng, chúng ta cần xây dựng $\mu$ sao cho để tác động của việc thay đổi nước hoặc bóng phụ thuộc vào giá trị của biến còn lại. Ví dụ, nếu nước thấp, thì giảm bóng không giúp gì nhiều như khi nước là cao. Chúng ta muốn slope của nước, $\beta_W$, được đặt điều kiện trên bóng. Tương tự với bóng được đặt điều kiện trên nước (nhớ lại tương tác của Buridan). Làm sao để chúng ta làm được điều đó?

Trong ví dụ trước, độ gồ ghề địa hình, chúng ta tạo một slope được đặt điều kiện trên giá trị của một nhóm. Khi theo nguyên tắc, có vô số nhóm, thì việc này khó hơn. Trong trường hợp này, "nhóm" của bóng và nước, theo nguyên tắc, vô hạn và có thứ tự. Chúng ta chỉ quan sát ba mức độ của nước. nhưng mô hình nên có thể cho dự đoán với mức độ trung gian của nước giữa bất kỳ hai mức độ quan sát được. Với tương tác liên tục, vấn đề không phải ở phần vô hạn mà là ở phần có thứ tự. Ngay cả nếu chúng ta chỉ quan tâm đến ba giá trị quan sát được, bạn vẫn muốn giữ tính thứ tự đó, tức là cái nào lớn hơn cái nào. Vậy làm gì tiếp?

Một đáp án theo quy ước là tái áp dụng thuyết địa tâm gốc để xác định hồi quy tuyến tính. Khi chúng ta có hai biến, biến kết cục và biến dự đoán, và chúng ta muốn mô hình hoá trung bình của kết cục để nó được đặt điều kiện trên giá trị của muốn biến dự đoán liên tục $x$, chúng ta có thể dùng mô hình tuyến tính: $\mu_i = \alpha + \beta x_i$. Bây giờ để làm cho slope $\beta$ được đặt điều kiện trên một biến khác nữa, chúng ta có thể áp dụng lần nữa bằng cách tương tự.

Để đơn giản, đặt $W_i$ và $S_i$ là các biến được canh giữa. Sau đó nếu chúng ta định nghĩa slope $B_W$ với mô hình tuyến tính $\gamma_W$ của riêng nó:

$$\begin{aligned}
\mu_i &= \alpha +\gamma_{W,i} W_i + \beta_S S_i\\
\gamma_{W,i} &= \beta_W + \beta_{WS} S_i\\
\end{aligned}$$

Bây giờ $\gamma_{W,i}$ là slope định nghĩa hoa thay đổi nhanh như thế nào với các mức độ nước. Tham số $\beta_W$ là tốc độ thay đổi, khi bóng ở giá trị trung bình của nó. Và $\beta_{WS}$ là tốc độ thay đổi trong $\gamma_{W,i}$ khi bóng thay đổi - slope của bóng trên slope của nước. Nhớ rằng, nó toàn là rùa ở phía dưới. Chú ý rằng $i$ trong $\gamma_{W,i}$ - nó phụ thuộc vào hàng $i$, bởi vì nó có $S_i$ trong nó.

Chúng ta cũng muốn cho phép quan hệ giữa bóng, $\beta_S$, phụ thuộc vào nước. May mắn thay, bởi vì tính đối xứng của tương tác đơn giản, chúng ta được miễn phí điều này. Không có cách nào để cụ thể hoá một tương tác tuyến tính đơn giản mà bạn có thể nói rằng hiệu ứng của một biến $x$ phụ thuộc $z$ nhưng hiệu ứng của $z$ lại không phụ thuộc $x$. Tôi sẽ giải thích điều này chi tiết hơn ở phần thông tin thêm. Tác động của điều này là chúng ta có thể thay thế $\gamma_{W,i}$ thành mệnh đề cho $\mu_i$:

$$\mu_i = \alpha + \underbrace{(\beta_W + \beta_{WS} S_i )}_{\gamma_{W,i}}W_i + \beta_S S_i = \alpha + \beta_W W_i + \beta_S S_i + \beta_{WS}S_iW_i $$

Tôi chỉ phân phối $W_i$ và đặt số hạng $S_iW_i$ vào cuối. Và nó là dạng quy ước cuối tương tác liên tục, với một số hạng thêm vào ở bên phải ngoài cùng chứa tích của hai biến.

Hãy đặt nó vào tulips. Mô hình tương tác là:

$$\begin{aligned}
\beta_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta_W W_i + B_SS_i + \beta_{WS}W_iS_i\\
\end{aligned}$$

Việc cuối cùng là chúng ta cần prior cho tham số tương tác mới, $\beta_{WS}$. Điều này khó, bởi vì tham số epicycle không có ý nghĩa tự nhiên rõ ràng. Nhưng, dự đoán được suy ra có thể giúp đỡ. Giả sử tương tác phù hợp mạnh nhất là đủ lớn để bóng làm cho nước không có hiệu ứng. Tức là:

$$ \gamma_{\beta_W} + \beta_{WS}S_i =0 $$

Nếu chúng ta đặt $S_i =1$ (cực đại trong ví dụ này), thì điều này nghĩa là sự tương tác cần phải có cùng mức độ như hiệu ứng chính, nhưng ngược lại: $\beta_{WS} = -\beta_W$. Nó là sự tương tác lớn nhất mà có thể hiểu được. Cho nên nếu chúng ta đặt prior của $\beta_{WS}$ có cùng độ lệch chuẩn bới $beta_W$, có thể điều đó không lố bịch. Tất cả những thứ này có dạng code:

<b>code 8.24</b>
```python
def model(water_cent, shade_cent, blooms_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 0.25))
    bw = numpyro.sample("bw", dist.Normal(0, 0.25))
    bs = numpyro.sample("bs", dist.Normal(0, 0.25))
    bws = numpyro.sample("bws", dist.Normal(0, 0.25))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bw * water_cent + bs * shade_cent + bws * water_cent * shade_cent
    numpyro.sample("blooms_std", dist.Normal(mu, sigma), obs=blooms_std)
m8_5 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_5,
    optim.Adam(1),
    Trace_ELBO(),
    shade_cent=d.shade_cent.values,
    water_cent=d.water_cent.values,
    blooms_std=d.blooms_std.values,
)
p8_5, losses = svi.run(random.PRNGKey(0), 1000)
```

Và nó là cấu trúc của sự tương tác liên tục đơn giản. Bạn có thể kiểm tra kết quả bằng `print_summary`. Bạn sẽ thấy `bws` là số âm. Điều này nói lên gì ở thang đo của kết cục? Nó thực ra không dễ để tưởng tượng từ chỉ tham số, đặc biệt khi giá trị của biến dự đoán đều có số dương và số âm.

Tiếp theo, hãy tìm ra cách minh hoạ những quái vật này.

<div class="alert alert-dark">
<p><strong>Tương tác hình thành như thế nào?></strong> Như trong bài chính, nếu bạn thay thế $\gamma_{W,i}$ vào $\mu_i$ trên và mở rộng:</p>
$$\mu_i = \alpha +(\beta_W +\beta_{WS}S_i)W_i +\beta_SS_i = \alpha +\beta_WW_i +\beta_SS_i + \beta_{WS}S_iW_i$$
<p>Bây giờ có thể lấy nhân tử chung để tạo thành $\gamma_{S,i}$ là quan hệ giữa bóng và hoa phụ thuộc vào nước:</p>
$$\begin{aligned}
\mu_i &= \alpha +\beta_W W_i + \gamma_{S,i}S_i\\
\gamma_{S,i} &= \beta_S + \beta_{SW}W_i\\
\end{aligned}$$
<p>Cho nên cả hai cách diễn giải đều đồng thời là đúng. Bạn có thể đặt cả hai định nghĩa $\gamma$ vào $\mu$ cùng lúc:</p>
$$\begin{aligned}
\mu_i &= \alpha + \gamma_{W,i}W_i + \gamma_{S,i}S_i\\
\gamma_{W,i} &= \beta_W + \beta_{WS}S_i\\
\gamma_{S,i} &= \beta_S + \beta_{SW}W_i\\
\end{aligned}$$
<p>Chú ý rằng tôi định nghĩa hai tham số tương tác khác nhau: $\beta_{WS}$ và $\beta_{SW}$. Bây giờ thay thế định nghĩa $\gamma$ vào $\mu$ và lấy nhân tử chung:</p>
$$\begin{aligned}
\mu_i &=\alpha + (\beta_W + \beta_{WS}S_i)W_i + (\beta_S +\beta_{SW}W_i)S_i\\
&= \alpha + \beta_W W_i + \beta_S S_i + (\beta_{WS} + \beta_{SW})W_iS_i\\
\end{aligned}$$
<p>Thứ duy nhất chúng ta có thể định danh trong mô hình này là tổng $\beta_{WS} + \beta+{SW}$, cho nên thực ra tổng này là một tham số đơn độc (chiều không gian trong posterior). Nó là cùng một mô hình tương tác. Chúng ta chỉ không thể phân biệt giữa nước phụ thuộc bóng và bóng phụ thuộc nước.</p>
<p>Một nguyên tắc khác để xây dựng $\mu_i$ là bắt đầu bằng đạo hàm $\delta \mu_i /\delta W_i = \beta_W + \beta_{WS}S_i$ và $\delta mu_i/\delta S_i = \beta_S + \beta_{WS}W_i$. Tìm một hàm số $\mu_i$ mà có thể thoả mãn cả hai cho ra mô hình truyền thống. Bằng cách thêm điều kiện biên và những kiến thức prior khác, bạn có thể sử dụng chiến thuật này để tìm hàm số hoa mỹ hơn. Nhưng đạo hàm có thể khó hơn. Cho nên bạn sẽ muốn được tư vấn bởi một nhà toán học thân thiện trong trường hợp này.</p>
</div>

### 8.3.3 Biểu đồ dự đoán posterior

Golem (mô hình) có sức mạnh kinh ngạc để lý luận, nhưng kỹ năng con người rất tệ. Golem cung cấp phân phối posterior của tính phù hợp cho những kết hợp các tham số. Nhưng với con người chúng ta để hiểu những kết luận suy ra từ nó, chúng ta cần giải mã posterior thành một thứ gì đó khác. Biển dự đoán được canh giữa hay không, việc vẽ dự đoán posterior luôn nói cho bạn biết những gì golem đang nghĩ, trên thang đo của kết cục. Đó là lý do chúng ta nhấn mạnh việc vẽ biểu đồ. Nhưng trong những chương trước, không có sự tương tác. Kết quả là, khi vẽ dự đoán mô hình như là hàm số của một biến bất kỳ, bạn có thể giữ những biến khác hằng định ở bất kỳ giá trị bạn muốn. Cho nên lựa chọn giá trị nào để đặt cho biến không được xem xét là không quan trọng.

Bây giờ sẽ khác. Một khi có tương tác trong mô hình, hiệu ứng của việc thay đổi biến dự đoán phụ thuộc vào giá trị của biến dự đoán khác. Có thể cách đơn giản nhất đẽ vẽ biểu đồ sự phụ thuộc lẫn nhau này là tạo ra một khung chứa nhiều biểu đồ hai biến. Trong mỗi biểu đồ, bạn chọn giá trị khác nhau cho những biến không được xem xét. Sau đó so sánh những biểu đồ với nhau, bạn có thể xem được sự khác nhau lớn như thế nào khi có sự thay đổi.

Đó là những gì chúng ta đã làm với ví dụ độ gồ ghề địa hình. Nhưng ở đó chúng chỉ cần hai biểu đồ, một cho Châu Phi và một cho những nơi khác. Bây giờ chúng ta sẽ cần nhiều hơn. Đây là cách bạn có thể vẽ được biểu đồ cho data tulip. Tôi sẽ tạo ra ba biểu đồ trong một bảng. Bảng có ba biểu đồ được xem cùng một lúc gọi là **TRIPTYCH**, và biểu đồ triptych là hữu ích để hiểu tác động của sự tương tác. Đây là các bước thực hiện. Chúng ta muốn mỗi biểu đồ thể hiện quan hệ hai biến giữa nước và hoa, theo dự đoán của mô hình. Mỗi biểu đồ sẽ vẽ dự đoán cho giá trị khác nhau của bóng. Ví dụ, bạn dễ chọn ra ba giá trị bóng để sử dụng, bởi vì có ba giá trị: -1, 0 và 1. Nhưng tổng quát hơn, bạn có thể chọn đại diện một giá trị thấp, trung bình, và một giá trị cao.

Đây là mã để vẽ phân phối posterior cho `m8_4`, mô hình không tương tác. Nó sẽ lặp lại ba giá trị của bóng, tính dự đoán posterior, sau đó vẽ 20 đường thẳng từ posterior.

<b>code 8.25</b>
```python
_, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)  # 3 plots in 1 row
for ax, s in zip(axes, range(-1, 2)):
    idx = d.shade_cent == s
    ax.scatter(d.water_cent[idx], d.blooms_std[idx])
    ax.set(xlim=(-1.1, 1.1), ylim=(-0.1, 1.1), xlabel="water", ylabel="blooms")
    post = m8_4.sample_posterior(random.PRNGKey(1), p8_4, (1000,))
    mu = Predictive(m8_4.model, post, return_sites=["mu"])(
        random.PRNGKey(2), shade_cent=s, water_cent=jnp.arange(-1, 2)
    )["mu"]
    for i in range(20):
        ax.plot(range(-1, 2), mu[i], "k", alpha=0.3)
```

<a name="f7"></a>![](/assets/images/fig 8-7.svg)
<details class="fig"><summary>Hình 8.7: Biểu đồ triptych cho hoa được dự đoán posterior xuyên suốt các điều trị nước và bóng. Hàng trên: Không có tương tác giữa nước và bóng. Hàng dưới: Với tương tác giữa nước và bóng. Mỗi biểu đồ có 20 đường thẳng posterior với mỗi mức độ của bóng.</summary>
{% highlight python %}_, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for row, m, p, name in zip(axes, [m8_4, m8_5], [p8_4, p8_5], ['m8_4','m8_5']):
    for ax, s in zip(row, range(-1, 2)):
        idx = d.shade_cent == s
        ax.scatter(d.water_cent[idx], d.blooms_std[idx])
        ax.set(xlim=(-1.1, 1.1), ylim=(-0.1, 1.1), xlabel="water", ylabel="blooms")
        post = m.sample_posterior(random.PRNGKey(1), p, (1000,))
        mu = Predictive(m.model, post, return_sites=["mu"])(
            random.PRNGKey(2), shade_cent=s, water_cent=jnp.arange(-1, 2)
        )["mu"]
        for i in range(20):
            ax.plot(range(-1, 2), mu[i], 'C0',alpha=0.3)
        ax.set(title=f"{name} post: shade={s}"){% endhighlight %}</details>

Kết quả được hiển thị ở [**HÌNH 8.7**](#f7), cùng với mô hình tương tác, `m8_5`. Chú ý rằng mô hình ở trên tin rằng nước có ích - slope dương trong mỗi biểu đồ - và bóng gây hại - những đường thẳng di chuyển xuống từ trái sang phải. Nhưng slope nước không thay đổi nhiều giữa các mức độ bóng. Khi không có sự tương tác, nó không thay đổi. Ở hàng dưới, sự tương tác được bật lên. Bây giờ mô hình tin rằng hiệu ứng của nước giảm đi khi bóng tăng lên. Nhừng đường thẳng nằm ngang dần.

Chuyện gì đã xảy ra? Lời giải thích phù hợp cho kết quả này là tulip cần cả nước và ánh sáng để tạo ra hoa. Ở điều kiện ánh sáng thấp, nước không thể có đủ hiệu ứng, bởi vì tulip không đủ ánh sáng để tạo ra hoa. Ở mức độ ánh sáng cao hơn, ánh sáng không còn ràng buộc việc nở hoa, và nên nước có đủ tác động tích cực lên kết cục. Giải thích tương tự cho tính đối xứng theo mức độ bóng. Nếu không có đủ ánh sáng, thì nhiều nước hơn cũng không có ích. Bạn có thể tạo lại [**HÌNH 8.7**](#f7) với bóng ở trục hoành và mức độ nước thay đổi từ trái sang phải, nếu bạn muốn minh hoạ dự đoán mô hình theo cách đó.

### 8.3.4 Biểu đồ dự đoán prior

Và chúng ta có thể sử dụng kỹ thuật tương tự để vẽ mô phỏng dự đoán prior. Điều này sẽ cho chúng ta cách lượng giá ước đoán từ trước. Để tạo ra dự đoán prior, tất cả những gì cần là trích xuất prior:

<b>code 8.26</b>
```python
predictive = Predictive(
    m8_5.model, num_samples=1000, return_sites=["a", "bw", "bs", "bws", "sigma"]
)
prior = predictive(random.PRNGKey(7), water_cent=0, shade_cent=0)
```

Tôi cũng chỉnh lại khoảng trục tung của biểu đồ prior, cho nên chúng ta có thể nhìn dễ hơn những đường thẳng ngoài khoảng kết cục phù hợp.

<a name="f8"></a>![](/assets/images/fig 8-8.svg)
<details class="fig"><summary>Hình 8.8: Biểu đồ triptych cho hoa nở được dự đoán prior xuyên suốt các điều trị nước và bóng. Hàng trên: Không có tương tác giữa nước và bóng. Hàng dưới: Với tương tác giữa nước và bóng. Mỗi biểu đồ cho 20 đường thẳng prior với mỗi mức độ bóng.</summary>
{% highlight python %}_, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for row, m, p, name in zip(axes, [m8_4, m8_5], [p8_4, p8_5], ['m8_4','m8_5']):
    for ax, s in zip(row, range(-1, 2)):
        idx = d.shade_cent == s
        ax.set(xlim=(-1.1, 1.1), ylim=(-0.1, 1.1), xlabel="water", ylabel="blooms")
        post = m.sample_posterior(random.PRNGKey(1), p, (1000,))
        pred = Predictive(m.model, num_samples=1000)(
            random.PRNGKey(2), shade_cent=s, water_cent=jnp.arange(-1, 2))
        mu = pred["mu"]
        for i in range(20):
            ax.plot(range(-1, 2), mu[i], 'C0',alpha=0.3)
        ax.set(title=f"{name} prior: shade={s}")
        ax.plot(range(-1,2), mu[22], "C1" ){% endhighlight %}</details>

Kết quả hiện thị ở [**HÌNH 8.8**](#f8). Bởi vì những đường thẳng được phân tán trong prior - prior không có nhiều thông tin - khó nhìn ra rằng những đường thẳng từ chung một tập mẫu thực ra hợp lại một cách có ý nghĩa. Cho nên tôi đã tô đậm ba đường thẳng ở hàng trên và hàng dưới. Cả ba đường thẳng tô đậm ở hàng trên đến từ chung một tham số. Chú ý rằng cả ba đều có chung một slope. Đây là thứ chúng ta mong đợi từ mô hình không có tương tác. Cho nên trong khi những đường thẳng trong prior có nhiều slope khác nhau, slope của nước không phụ thuộc vào bóng. Ở hàng dưới, ba đường thẳng tô đậm lần nữa đến từ chung một mẫu prior. Nhưng bây giờ sự tương tác làm cho slope thay đổi có hệ thống khi mức độ bóng thay đổi.

Chúng ta có thể nói gì với prior này, sau tất cả? Chúng là vô hại, nhưng tính thực dụng yếu. Đa số những đường thẳng nằm trong không gian kết cục phù hợp. Nhưng những xu hướng ngu ngốc là không hiếm. Chúng ta có thể làm tốt hơn. Chúng ta có thể làm tệ hơn, nhưng prior phẳng mà xem xét ngay cả khi tăng một ít bóng sẽ giết tất cả tulip. Nếu bạn thể hiện những prior này cho đồng nghiệp, một tóm tắt hợp lý có thể là, "Những prior này không chứa sai lệch nào về hiệu ứng dương hoặc âm, và đồng thời chúng ràng buộc yếu những hiệu ứng vào khoảng thực tế."

## <center>8.4 Tổng kết</center><a name="a4"></a>

Chương này giới thiệu *sự tương tác*, cho phép quan hệ giữa biến dự đoán và biến kết cục phụ thuộc vào giá trị của biến dự đoán khác. Trong khi bạn không thể nhìn thấy chúng trong DAG, sự tương tác có thể là quan trọng để thực hiện suy luận chính xác. Sự tương tác có thể khó diễn giải, và cho nên chương này cũng giới thiệu biểu đồ *triptych* để giúp minh minh hoạ hiệu ứng của sự tương tác. Không có kỹ năng code mới nào được giới thiệu, nhưng mô hình thống kê được nói đến là một trong những mô hình phức tạp nhất đến giờ trong sách này. Để đi xa hơn, bạn cần phải có một cỗ máy đặt điều kiện mạnh mẽ hơn để fit mô hình vào data. Nó là chủ đề trong chương tiếp theo.

---

<details><summary>Endnotes</summary>
<ol class="endnotes">
<li><a name="135" href="#r135">135. </a>All manatee facts here taken from Lightsey et al. (2006); Rommel et al. (2007). Scar chart in figure from the free educational materials at http://www.learner.org/jnorth/tm/manatee/RollCall.html.</li>
<li><a name="136" href="#r136">136. </a>Wald (1943). See Mangel and Samaniego (1984) for a more accessible presentation and historical context.</li>
<li><a name="137" href="#r137">137. </a>Wald (1950). Wald’s foundational paper is Wald (1939). Fienberg (2006) is a highly recommended read for historical context. For more technical discussions, see Berger (1985), Robert (2007), and Jaynes (2003) page 406.</li>
<li><a name="138" href="#r138">138. </a>GDP is Gross Domestic Product. It’s the most common measure of economic performance, but also one of the silliest. Using GDP to measure the health of an economy is like using heat to measure the quality of a chemical reaction.</li>
<li><a name="139" href="#r139">139. </a>Riley et al. (1999).</li>
<li><a name="140" href="#r140">140. </a>From Nunn and Puga (2012).</li>
<li><a name="141" href="#r141">141. </a>A good example is the extensive modern tunnel system in the Faroe Islands. The natural geology of the islands is very rugged, such that it has historically been much easier to travel by water than by land. But in the late twentieth century, the Danish government invested heavily in tunnel construction, greatly reducing the effective ruggedness of the islands.</li>
<li><a name="142" href="#r142">142. </a>Modified example from Grafen and Hails (2002), which is a great non-Bayesian applied statistics book you might also enjoy. It has a rather unique geometric presentation of some of the standard linear models.</li>
<li><a name="143" href="#r143">143. </a>Data from Nettle (1998).</li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
<p><strong>8E1.</strong> For each of the causal relationships below, name a hypothetical third variable that would lead to an interaction effect.</p>
<ol>
<li>Bread dough rises because of yeast.</li>
<li>Education leads to higher income.</li>
<li>Gasoline makes a car go.</li>
</ol>
<p><strong>8E2.</strong> Which of the following explanations invokes an interaction?</p>
<ol>
<li>Caramelizing onions requires cooking over low heat and making sure the onions do not dry out.</li>
<li>A car will go faster when it has more cylinders or when it has a better fuel injector.</li>
<li>Most people acquire their political beliefs from their parents, unless they get them instead from their friends.</li>
<li>Intelligent animal species tend to be either highly social or have manipulative appendages (hands, tentacles, etc.).</li>
</ol>
<p><strong>8E3.</strong> For each of the explanations in 8E2, write a linear model that expresses the stated relationship.</p>
<p><strong>8M1.</strong> Recall the tulips example from the chapter. Suppose another set of treatments adjusted the temperature in the greenhouse over two levels: cold and hot. The data in the chapter were collected at the cold temperature. You find none of the plants grown under the hot temperature developed any blooms at all, regardless of the water and shade levels. Can you explain this result in terms of interactions between water, shade, and temperature?</p>
<p><strong>8M2.</strong> Can you invent a regression equation that would make the bloom size zero, whenever the temperature is hot?</p>
<p><strong>8M3.</strong> In parts of North America, ravens depend upon wolves for their food. This is because ravens are carnivorous but cannot usually kill or open carcasses of prey. Wolves however can and do kill and tear open animals, and they tolerate ravens co-feeding at their kills. This species relationship is generally described as a “species interaction.” Can you invent a hypothetical set of data on raven population size in which this relationship would manifest as a statistical interaction? Do you think the biological interaction could be linear? Why or why not?</p>
<p><strong>8M4.</strong> Repeat the tulips analysis, but this time use priors that constrain the effect of water to be positive and the effect of shade to be negative. Use prior predictive simulation. What do these prior assumptions mean for the interaction prior, if anything?</p>
<p><strong>8H1.</strong> Return to the data(tulips) example in the chapter. Now include the <code>bed</code> variable as a predictor in the interaction model. Don’t interact <code>bed</code> with the other predictors; just include it as a main effect. Note that <code>bed</code> is categorical. So to use it properly, you will need to either construct dummy variables or rather an index variable, as explained in Chapter 5.</p>
<p><strong>8H2.</strong> Use WAIC to compare the model from <strong>8H1</strong> to a model that omits <code>bed</code>. What do you infer from this comparison? Can you reconcile the WAIC results with the posterior distribution of the <code>bed</code> coefficients?</p>
<p><strong>8H3.</strong> Consider again the data(rugged) data on economic development and terrain ruggedness, examined in this chapter. One of the African countries in that example, Seychelles, is far outside the cloud of other nations, being a rare country with both relatively high GDP and high ruggedness. Seychelles is also unusual, in that it is a group of islands far from the coast of mainland Africa, and its main economic activity is tourism.</p>
<ol type='a'>
<li>Focus on model <code>m8_5</code> from the chapter. Use WAIC pointwise penalties and PSIS Pareto $k$ values to measure relative influence of each country. By these criteria, is Seychelles influencing the results? Are there other nations that are relatively influential? If so, can you explain why?</li>
<li>Now use robust regression, as described in the previous chapter. Modify <code>m8_5</code> to use a Student-t distribution with $\nu = 2$. Does this change the results in a substantial way?</li>
</ol>
<p><strong>8H4.</strong> The values in data(nettle) are data on language diversity in 74 nations.<sup><a name="r143" href="#143">143</a></sup> The meaning of each column is given below.</p>
<ol type='a'>
<li><code>country</code>: Name of the country</li>
<li><code>num.lang</code>: Number of recognized languages spoken</li>
<li><code>area</code>: Area in square kilometers</li>
<li><code>k.pop</code>: Population, in thousands</li>
<li><code>num.stations</code>: Number of weather stations that provided data for the next two columns</li>
<li><code>mean.growing.season</code>: Average length of growing season, in months</li>
<li><code>sd.growing.season</code>: Standard deviation of length of growing season, in months</li>
</ol>
<p>Use these data to evaluate the hypothesis that language diversity is partly a product of food security. The notion is that, in productive ecologies, people don’t need large social networks to buffer them against risk of food shortfalls. This means cultural groups can be smaller and more self-sufficient, leading to more languages per capita. Use the number of languages per capita as the outcome:</p>
<b>code 8.27</b>
{% highlight python %}d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/nettle.csv?raw=true", sep=";")
d["lang.per.cap"] = d["num.lang"] / d["k.pop"]{% endhighlight %}
<p>Use the logarithm of this new variable as your regression outcome. (A count model would be better here, but you’ll learn those later, in Chapter 11.) This problem is open ended, allowing you to decide how you address the hypotheses and the uncertain advice the modeling provides. If you think you need to use WAIC anyplace, please do. If you think you need certain priors, argue for them. If you think you need to plot predictions in a certain way, please do. Just try to honestly evaluate the main effects of both <code>mean.growing.season</code> and <code>sd.growing.season</code>, as well as their two-way interaction. Here are three parts to help. (a) Evaluate the hypothesis that language diversity, as measured by <code>log(lang.per.cap)</code>, is positively associated with the average length of the growing season, <code>mean.growing.season</code>. Consider <code>log(area)</code> in your regression(s) as a covariate (not an interaction). Interpret your results. (b) Now evaluate the hypothesis that language diversity is negatively associated with the standard deviation of length of growing season, <code>sd.growing.season</code>. This hypothesis follows from uncertainty in harvest favoring social insurance through larger social networks and therefore fewer languages. Again, consider <code>log(area)</code> as a covariate (not an interaction). Interpret your results. (c) Finally, evaluate the hypothesis that <code>mean.growing.season</code> and sd.growing.season interact to synergistically reduce language diversity. The idea is that, in nations with longer average growing seasons, high variance makes storage and redistribution even more important than it would be otherwise. That way, people can cooperate to preserve and protect windfalls to be used during the droughts.</p>
<p><strong>8H5.</strong> Consider the data(Wines2012) data table. These data are expert ratings of 20 different French and American wines by 9 different French and American judges. Your goal is to model <code>score</code>, the subjective rating assigned by each judge to each wine. I recommend standardizing it. In this problem, consider only variation among judges and wines. Construct index variables of <code>judge</code> and <code>wine</code> and then use these index variables to construct a linear regression model. Justify your priors. You should end up with 9 judge parameters and 20 wine parameters. How do you interpret the variation among individual judges and individual wines? Do you notice any patterns, just by plotting the differences? Which judges gave the highest/lowest ratings? Which wines were rated worst/best on average?</p>
<p><strong>8H6.</strong> Now consider three features of the wines and judges:</p>
<ol type='a'>
<li><code>flight</code>: Whether the wine is red or white.</li>
<li><code>wine.amer</code>: Indicator variable for American wines.</li>
<li><code>judge.amer</code>: Indicator variable for American judges.</li>
</ol>
<p>Use indicator or index variables to model the influence of these features on the scores. Omit the individual judge and wine index variables from Problem 1. Do not include interaction effects yet. Again justify your priors. What do you conclude about the differences among the wines and judges? Try to relate the results to the inferences in the previous problem.</p>
<p><strong>8H7.</strong> Now consider two-way interactions among the three features. You should end up with three different interaction terms in your model. These will be easier to build, if you use indicator variables. Again justify your priors. Explain what each interaction means. Be sure to interpret the model’s predictions on the outcome scale (<code>mu</code>, the expected score), not on the scale of individual parameters. You can use <code>Predictive</code> to help with this, or just use your knowledge of the linear model instead. What do you conclude about the features and the scores? Can you relate the results of your model(s) to the individual judge and wine inferences from <strong>8H5</strong>?</p>
</details>