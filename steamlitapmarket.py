import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules,apriori

st.set_page_config(
    page_title="MarketBasket Apriori",
    page_icon=":basket:",
    layout="wide"
)
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #FFC785;
    }
</style>

<style>[data-testid="stAppViewContainer"] {
        background-image: url("https://blog.1a23.com/wp-content/uploads/sites/2/2020/02/pattern-5.svg"),
            linear-gradient(#4d4d4d, transparent),
            linear-gradient(to top left, #333333, transparent),
            linear-gradient(to top right, #4d4d4d, transparent);
        background-size: contain;
        width: 100%;
        height: 100vh;
        position: fixed;
        background-position: left;
        background-repeat: repeat-x;
        background-blend-mode: darken;
        will-change: transform;
    }</style>
""", unsafe_allow_html=True)

with st.sidebar:
    "## WEB APP by"
    "## --"
""""""

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")
st.write("""
    Anda dapat mengunjungi 
    [Kaggle](https://www.kaggle.com/datasets/mittalvasu95/the-bread-basket) 
    untuk melihat dataset lebih detail.
    """)
st.sidebar.header("Tentang")
st.sidebar.success("Algoritma apriori adalah sebuah algoritma pencarian pola yang sangat populer dalam teknik penambangan data (datamining). Algoritma ini ditujukan untuk mencari kombinasi item-set yang mempunyai suatu nilai keseringan tertentu sesuai kriteria atau filter yang diinginkan. Hasil dari algoritma ini dapat digunakan untuk membantu dalam pengambilan keputusan pihak manajemen.")
st.sidebar.success("Market basket analysis (MBA) adalah teknik analisis yang mempelajari pola pembelian pelanggan untuk menemukan hubungan antara produk yang sering dibeli bersama. Teknik ini merupakan bagian dari ilmu data mining")
    
""""""
#load dataset 
df = pd.read_csv("bread_basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace=True)


def get_data(period_day = '', weekday_weekend = '', month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data ["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "NO Result!"

def user_input_freatures():
    item = st.selectbox("Item", ['Bread', 'Scandinavian', 'Hot chocolate', 'Jam', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'Tartine', 'Basket', 'Mineral water', 'Farm House', 'Fudge', 'Juice', "Ella's Kitchen Pouches", 'Victorian Sponge', 'Frittata', 'Hearty & Seasonal', 'Soup', 'Pick and Mix Bowls', 'Smoothies', 'Cake', 'Mighty Protein', 'Chicken sand', 'Coke', 'My-5 Fruit Shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs', 'Brownie', 'Dulce de Leche', 'Honey', 'The BART', 'Granola', 'Fairy Doors', 'Empanadas', 'Keeping It Local', 'Art Tray', 'Bowl Nic Pitt', 'Bread Pudding', 'Adjustment', 'Truffles', 'Chimichurri Oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings', 'Caramel bites', 'Jammie Dodgers', 'Tiffin', 'Olum & polenta', 'Polenta', 'The Nomad', 'Hack the stack', 'Bakewell', 'Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie', 'Bare Popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup', 'Panatone', 'Brioche and salami', 'Afternoon with the baker', 'Salad', 'Chicken Stew', 'Spanish Brunch', 'Raspberry shortbread sandwich', 'Extra Salami or Feta', 'Duck egg', 'Baguette', "Valentine's card", 'Tshirt', 'Vegan Feast', 'Postcard', 'Nomad bag', 'Chocolates', 'Coffee granules ', 'Drinking chocolate spoons ', 'Christmas common', 'Argentina Night', 'Half slice Monster ', 'Gift voucher', 'Cherry me Dried fruit', 'Mortimer', 'Raw bars', 'Tacos/Fajita'])
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sat")

    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_freatures()

data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(data) != type ("No Result"):
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric,min_threshold=min_threshold)[["antecedents","consequents","support","confidence","lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0,:])

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika Konsumen Membeli **{item}**, Maka Membeli **{return_item_df(item)[1]}** secara bersamaan")
