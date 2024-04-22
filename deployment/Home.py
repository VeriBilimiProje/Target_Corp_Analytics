import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('datasets/final_dataset.csv')

st.set_page_config(
    page_title="Data Sapiens",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

html_code = """
<div style="display: flex;">
    <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://media.licdn.com/dms/image/D4D03AQGcvyGbq29esQ/profile-displayphoto-shrink_400_400/0/1713544700654?e=1719446400&v=beta&t=8rNFjSu46qxavynGcNQTUXZ4kDO7ewEf_TYxViYLi5s" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Muhammed AKAR</div>
    </a>
    <a href="https://github.com/umitdkara" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/154842224?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Ümit KARA</div>
    </a>
    <a href="https://github.com/ecan57" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/105751954?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Emine CAN</div>
    </a>
    <a href="https://github.com/leylalptekin" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/48180024?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Leyla ALPTEKİN</div>
    </a>
</div>

"""

st.title('DATA SAPIENS')
st.markdown(html_code, unsafe_allow_html=True)
'---'
st.subheader('Grup Marşımız')
st.audio("deployment/assets/datasapiens.mp3", format="audio/mpeg")
'---'


st.subheader('Proje Hakkında')

st.markdown('''
**Giriş:**

Bu, Olist Store'da yapılan siparişlerin Brezilya e-ticaret genel veri kümesidir. Veri kümesi, 2016'dan 2018'e kadar Brezilya'daki birden fazla pazar yerinde yapılan 100 bin siparişin bilgisine sahiptir. Özellikleri, bir siparişi birden fazla boyuttan görüntülemeye olanak tanır: sipariş durumu, fiyat, ödeme ve nakliye performansından müşteri konumuna, ürün özelliklerine ve son olarak müşteriler tarafından yazılan incelemelere kadar. Ayrıca Brezilya posta kodlarını enlem/boylam koordinatlarıyla ilişkilendiren bir coğrafi konum veri seti de yayınladık.
Bu gerçek ticari veridir, anonimleştirilmiştir ve inceleme metnindeki şirketlere ve ortaklara yapılan referanslar Game of Thrones'un büyük evlerinin isimleriyle değiştirilmiştir.


**Müşteri Memnuniyeti Modelleme:**

Müşteri memnuniyeti modelleme, müşteri memnuniyetini etkileyen faktörleri belirleme ve bu faktörlere dayalı olarak müşteri memnuniyetini tahmin etme sürecidir. Bu modeller, pazarlama kampanyalarını optimize etmek, ürün ve hizmetleri geliştirmek ve müşteri deneyimini iyileştirmek için kullanılabilir.

**Olist Veri Seti:**

Olist veri seti, müşteri memnuniyeti modelleme için ideal bir kaynaktır. Veri seti, siparişler, ürünler, müşteriler ve değerlendirmeler gibi müşteri memnuniyetini etkileyebilecek birçok faktörü içerir.

**Modelleme Yöntemleri:**

Müşteri memnuniyetini modellemek için çeşitli yöntemler kullanılabilir. Yaygın olarak kullanılan yöntemlerden bazıları şunlardır:

*   **Regresyon analizi:** Regresyon analizi, bağımlı bir değişken (örneğin, müşteri memnuniyeti puanı) ile bir veya birden fazla bağımsız değişken (örneğin, ürün fiyatı, teslimat süresi) arasındaki ilişkiyi modellemek için kullanılır.
    
*   **Sınıflandırma:** Sınıflandırma, müşterileri memnun veya memnuniyetsiz olarak sınıflandırmak için kullanılır.
    
*   **Makine öğrenmesi:** Makine öğrenmesi, müşteri memnuniyetini etkileyen karmaşık ilişkileri otomatik olarak öğrenmek için kullanılır.
    

**Modelleme Değerlendirme:**

Müşteri memnuniyeti modelleri, farklı ölçütler kullanılarak değerlendirilmelidir. Yaygın olarak kullanılan ölçütlerden bazıları şunlardır:

*   **Ortalama mutlak hata (MAE):** MAE, tahmin edilen değerler ile gerçek değerler arasındaki ortalama mutlak farkı ölçer.
    
*   **Ortalama karesel hata (MSE):** MSE, tahmin edilen değerler ile gerçek değerler arasındaki ortalama karesel farkı ölçer.
    
*   **Doğruluk:** Doğruluk, doğru bir şekilde tahmin edilen müşteri sayısının toplam müşteri sayısına oranını ölçer.
    

**Sonuç:**

Olist veri seti, müşteri memnuniyeti modelleme için değerli bir kaynaktır. Bu veri seti, siparişler, ürünler, müşteriler ve değerlendirmeler gibi çeşitli bilgileri içerir. Bu bilgiler, müşteri memnuniyetini etkileyen faktörleri belirlemek ve tahmin modelleri oluşturmak için kullanılabilir.
''')

'---'

st.subheader('Proje Dosyaları')

st.markdown('''
<div style="display: flex;">
    <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%; padding:10px;order: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-kaggle-3521526-2945029.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">Kaggle</div>
    </a>
       <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%;padding:10px;order: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-github-159-721954.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">GitHub</div>
    </a>
''',unsafe_allow_html=True)

