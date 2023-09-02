
##############################
# Students' dropout
##############################

# Problem : A dataset created from a higher education institution (acquired from several disjoint databases)
# related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism,
# management, social service, and technologies. The dataset includes information
# known at the time of student enrollment (academic path, demographics, and social-economic factors)
# and the students' academic performance at the end of the first and second semesters.
# The data is used to build classification models to predict students' dropout and academic success.
# The problem is formulated as a three category classification task, in which
# there is a strong imbalance towards one of the classes.

# Türkçe:
# Problem : Tarım bilimi, tasarım, eğitim, hemşirelik, gazetecilik, yönetim, sosyal hizmet ve teknolojiler gibi
# farklı lisans derecelerine kayıtlı öğrencilerle ilgili bir yüksek öğretim kurumundan oluşturulan
# (birkaç ayrı veri tabanından elde edilen) bir veri kümesi.
# Veri seti, öğrenci kaydı sırasında bilinen bilgileri (akademik yol, demografik bilgiler ve sosyo-ekonomik faktörler)
# ve öğrencilerin birinci ve ikinci dönem sonundaki akademik performansını içerir.
# Veriler, öğrencilerin okuldan ayrılmalarını ve akademik başarılarını
# tahmin etmek amacıyla sınıflandırma modelleri oluşturmak için kullanılır.
# Problem, sınıflardan birine karşı güçlü bir dengesizliğin olduğu
# üç kategorili bir sınıflandırma görevi olarak formüle edilmiştir.
# 37 Değişken 4424 Gözlem


# Amaç:For what purpose was the dataset created?

# The dataset was created in a project that aims to contribute to the reduction of academic dropout and failure in higher education,
# by using machine learning techniques to identify students at risk at an early stage of their academic path,
# so that strategies to support them can be put into place.
# The dataset includes information known at the time of student enrollment – academic path, demographics,
# and social-economic factors.
# The problem is formulated as a three category classification task
# (dropout, enrolled, and graduate) at the end of the normal duration of the course.

# Türkçe: Veri seti, yüksek öğrenimde akademik terk ve başarısızlığın azaltılmasına katkıda bulunmayı amaçlayan
# bir proje kapsamında, risk altındaki öğrencileri akademik yollarının erken bir aşamasında tespit etmek için makine öğrenimi
# tekniklerini kullanarak oluşturuldu; böylece onları destekleyecek stratejiler geliştirilebilir.
# Veri seti, öğrenci kaydı sırasında bilinen bilgileri (akademik yol, demografik bilgiler ve sosyo-ekonomik faktörler) içerir.
# Problem, kursun normal süresinin sonunda üç kategorili bir sınıflandırma görevi (bırakma, kaydolma ve mezun olma)
# olarak formüle edilmiştir.

# Veri seti hakkında detay bilgiler:
# Bu veri kümesindeki her örnek bir öğrencidir.
# Anormallikler, açıklanamayan aykırı değerler ve eksik değerler filtrelenmiştir.
#

# 37 Değişken 4424 Gözlem


# Marital status : Integer :  1 – single 2 – married 3 – widower (dul) 4 – divorced  5 – facto union 6 – legally separated
# Application mode : Integer :  Kontenjan grupları;	1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)
# Application order :Integer : Tercih/Başvuru sırası; (between 0 - first choice; and 9 last choice)
# Course : Integer : Hangi Bölüm/Kursa kayıtlı olduğu 33 - Biofuel Production Technologies 171 - Animation and Multimedia Design 8014 - Social Service (evening attendance) 9003 - Agronomy 9070 - Communication Design 9085 - Veterinary Nursing 9119 - Informatics Engineering 9130 - Equinculture 9147 - Management 9238 - Social Service 9254 - Tourism 9500 - Nursing 9556 - Oral Hygiene 9670 - Advertising and Marketing Management 9773 - Journalism and Communication 9853 - Basic Education 9991 - Management (evening attendance)
# Daytime/evening attendance : Integer : 1. öğretim mi 2. öğretim mi? 1 – daytime 0 - evening
# Previous qualification : Integer : Önceki mezuniyet durumu : 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle)
# Previous qualification (grade) : Float  DATASETDE VAR MAKALEDE YOK önceki mezuniyetleri sıralamaları--> Grade of previous qualification (between 0 and 200)
# Nacionality   : Integer Öğrencilerin kökenleri 	1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian
# Mother's qualification : Integer :  Annenin mezuniyet durumu : 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)
# Father's qualification  : Integer : Babanın mezuniyet durumu : 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 13 - 2nd year complementary high school course 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 20 - Complementary High School Course 22 - Technical-professional course 25 - Complementary High School Course - not concluded 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 31 - General Course of Administration and Commerce 33 - Supplementary Accounting and Administration 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)
# Mother's occupation  :  Integer : Annenin mesleği : 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants
# Father's occupation  :  Integer : Babanın mesleği : 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 101 - Armed Forces Officers 102 - Armed Forces Sergeants 103 - Other Armed Forces personnel 112 - Directors of administrative and commercial services 114 - Hotel, catering, trade and other services directors 121 - Specialists in the physical sciences, mathematics, engineering and related techniques 122 - Health professionals 123 - teachers 124 - Specialists in finance, accounting, administrative organization, public and commercial relations 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 135 - Information and communication technology technicians 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 154 - Protection and security services personnel 161 - Market-oriented farmers and skilled agricultural and animal production workers 163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence 171 - Skilled construction workers and the like, except electricians 172 - Skilled workers in metallurgy, metalworking and similar 174 - Skilled workers in electricity and electronics 175 - Workers in food processing, woodworking, clothing and other industries and crafts 181 - Fixed plant and machine operators 182 - assembly workers 183 - Vehicle drivers and mobile equipment operators 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants 195 - Street vendors (except food) and street service providers
# Admission grade : Float DATASETDE VAR MAKALEDE YOK Giriş Notu : Admission grade (between 0 and 200)
# Displaced : Integer : Farklı yerlerden gelenler	1 – yes 0 – no
# Educational special needs : Integer :  Özel eğitim öğrencilerini ifade ediyor. Engelli vb gibi  1 – yes 0 – no
# Debtor : Integer : Borcu var mı? Yok mu? 1 – yes 0 – no
# Tuition fees up to date  : Integer:  Ödemesini zamanında yapmış mı? Yapmamış mı?  1 – yes 0 – no
# Gender  : Integer : Cinsiyet 1 – male 0 – female
# Scholarship holder : Burs alıyor mu? 1 – yes 0 – no
# Age at enrollment : kayıt sırasındaki yaş
# International : Uluslararası bir öğrenci mi değil mi? 1 – yes 0 – no

# Curricular units 1st sem (credited) : Integer : "Number of curricular units credited in the 1st semester" ifadesi,
# bir öğrencinin birinci dönemde tamamladığı ders birimlerinin sayısını ifade eder.
# "Curricular units" terimi ders birimlerini temsil ederken, "credited" terimi de bu ders birimlerini başarıyla tamamladığı veya geçtiği anlamına gelir.
# Yani bu ifade, öğrencinin birinci dönemde başarılı bir şekilde tamamladığı ders birimi sayısını belirtir.

# Curricular units 1st sem (enrolled) : Integer : "Curricular units 1st sem (enrolled)" ifadesi,
# bir öğrencinin birinci dönemde kayıt yaptırdığı ancak henüz tamamlamadığı ders birimlerinin sayısını ifade eder.
# "Curricular units" terimi ders birimlerini temsil ederken, "(enrolled)" terimi de bu ders birimlerine kayıt yaptırıldığı ancak henüz tamamlanmadığı anlamına gelir.
# Yani bu ifade, öğrencinin birinci dönemde kayıt yaptırdığı ancak henüz tamamlamadığı ders birimi sayısını belirtir.

# Curricular units 1st sem (evaluations) : Integer : "Number of evaluations to curricular units in the 1st semester" ifadesi,
# birinci dönemdeki ders birimlerine yönelik yapılan değerlendirme sayısını ifade eder. Burada "curricular units" ders birimlerini,
# "number of evaluations" ise yapılan değerlendirme sayısını temsil eder.
# Bu ifade, öğrencinin birinci dönemdeki ders birimlerine kaç tane değerlendirme yapıldığını belirtir.
# Bu değerlendirmeler genellikle sınavlar, ödevler veya projeler gibi öğrencilerin başarılarını ölçmeye yönelik faaliyetleri içerebilir.

# Curricular units 1st sem (approved) : Integer : "Number of curricular units approved in the 1st semester" ifadesi,
# birinci dönemde onaylanmış/denetlenmiş ders birimi sayısını ifade eder.
# Burada "curricular units" ders birimlerini ve "number of approved" onaylanmış/denetlenmiş ders birimi sayısını temsil eder.
# Bu ifade, öğrencinin birinci dönemde başarılı bir şekilde tamamladığı veya geçtiği ders birimi sayısını belirtir.
# Yani, öğrencinin birinci dönemde başarılı bir şekilde tamamladığı ders birimlerinin sayısıdır.

# Curricular units 1st sem (grade) : Float : 1. dönem not ortalaması (0-20 arasında)

# Curricular units 1st sem (without evaluations) : "Number of curricular units without evaluations in the 1st semester" ifadesi,
# birinci dönemde değerlendirmeleri olmayan ders birimi sayısını ifade eder.
# Burada "curricular units" ders birimlerini, "without evaluations" ise değerlendirmesi olmayan ders birimleri anlamına gelir.
# Bu ifade, öğrencinin birinci dönemde değerlendirmeye tabi tutulmamış yani sınav veya değerlendirme yapılmamış ders birimi sayısını belirtir.

# Curricular units 2nd sem (credited) : 1. dönemdeki bağımsız değişken ile aynı
# Curricular units 2nd sem (enrolled) : 1. dönemdeki bağımsız değişken ile aynı
# Curricular units 2nd sem (evaluations) : 1. dönemdeki bağımsız değişken ile aynı
# Curricular units 2nd sem (approved) : 1. dönemdeki bağımsız değişken ile aynı
# Curricular units 2nd sem (grade) : 1. dönemdeki bağımsız değişken ile aynı
# Curricular units 2nd sem (without evaluations) : 1. dönemdeki bağımsız değişken ile aynı

# Unemployment rate : Float: İşsizlik oranı (yüzde 7  ile  16.2 arasında değişiyor) Makroekonomik data Ülkelere göre değil tam anlamadım?
# Inflation rate : Float: Enflasyon oranı (yüzde) - 0.8 ile 3.7 arasında değişiyor
# GDP : Float: Gayri safi yurtiçi hasıla değeri -4.060 ile 3.510 arasında değişiyor

# Target : object Bu problem, bir kursun normal süresinin sonunda (belirtilen sürenin bitiminde) üç kategoriye ayrılan bir sınıflandırma görevi olarak formüle edilmiştir:
# ayrılan süre içinde bırakanlar (dropout), kayıtlı olanlar (enrolled) ve mezun olanlar (graduate)."


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
# Adım 1: Genel resmi inceleyiniz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
# Adım 5: Aykırı gözlem analizi yapınız.
# Adım 6: Eksik gözlem analizi yapınız.
# Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# işlemleri uygulayabilirsiniz.
# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# Adım 5: Model oluşturunuz.


# Gerekli Kütüphane ve Fonksiyonlar

import numpy as np
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("C:/Users/mehmet kupeli/PycharmProjects/pythonProject/datasets/data.csv", engine='python', sep=None)
# Graduate olan ya da olmayanlar diye ikiye ayırdım
df['Target'] = df['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)
df.head()
df.shape
df.info()

# Verisetine StudentID eklendi

df = df.reset_index(drop=False)
Student_ID = 'Student_ID'
df.rename(columns={"index": Student_ID}, inplace=True)
df["Student_ID"] = df["Student_ID"] + 1

# Değişken isimleri büyütmek ve boşluk varsa ortadan kaldırmak
df.columns = [col.upper().replace(" ", "_") for col in df.columns]


# df matrisinden anlamsız gelen 1. dönem sonu ve 2. dönem sonu değerleri çıkabilir. Pek anlaşılmıyor. Şimdilik yapmadım.
# eliminated_columns = df.iloc[:, 19:31]
# df = df.drop(columns=eliminated_columns)

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=15, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # Threshld üstünde kalan ama kategorik olan değişkenleri manuel ekledim
    deltas = ['APPLICATION_MODE', 'PREVIOUS_QUALIFICATION', 'NACIONALITY', "MOTHER'S_QUALIFICATION",
              "FATHER'S_QUALIFICATION", "MOTHER'S_OCCUPATION", "FATHER'S_OCCUPATION", "COURSE"]

    cat_cols += [deltas for deltas in deltas]
    num_cols = list(set(num_cols) - set(deltas))
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

for col in cat_cols:
    cat_summary(df, col)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=False)

#####
# PCA Görselleştirme
#####
y = df["TARGET"]
X = df[num_cols]


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df


pca_df = create_pca_df(X, y)


def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


plot_pca(pca_df, "TARGET")


##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "TARGET", col)


##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "TARGET", col)


##################################
# KORELASYON
##################################

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)
df.corrwith(df["TARGET"]).sort_values(ascending=False)
df[num_cols].corr()

##################################
# Data Preprocessing & Feature Engineering
##################################
# EKSİK DEĞER ANALİZİ
df.isnull().sum()


# Outlier Analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col, 0.01, 0.99))


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


check_df(df)

cat_cols = [col for col in cat_cols if "TARGET" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=30)

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["TARGET"]
X = df.drop(["TARGET", "STUDENT_ID"], axis=1)

check_df(X)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


missing_values_table(df)

##################################
# BASE MODEL KURULUMU
##################################


cat_cols = [col for col in cat_cols if col not in ["TARGET"]]
cat_cols

check_df(df)
y = df["TARGET"]
X = df.drop(["TARGET", "STUDENT_ID"], axis=1)

#####################
# Holdout yöntemi Başarı Değerlendirme (LogisticRegression Modeli için)
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
LR1 = LogisticRegression(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = LR1.predict(X_train)
y_prob = LR1.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = LR1.predict(X_test)
y_prob = LR1.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

models = [('LR', LogisticRegression(random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          ('SVM', SVC(gamma='auto', random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(random_state=17))]

# CatBoost'da hata verdi. Çıkarttım sonra bakarız. "CatBoost", CatBoostClassifier(verbose=-1, random_state=12345)
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Model Başarıları
########## LR ##########
""" Accuracy: 0.8513
Auc: 0.9255
Recall: 0.8846
Precision: 0.8301
F1: 0.8561
########## KNN ##########
Accuracy: 0.7796
Auc: 0.8462
Recall: 0.8728
Precision: 0.7359
F1: 0.7982
########## CART ##########
Accuracy: 0.7869
Auc: 0.7869
Recall: 0.7963
Precision: 0.7816
F1: 0.7884
########## RF ##########
Accuracy: 0.847
Auc: 0.9222
Recall: 0.8791
Precision: 0.8264
F1: 0.8516
########## SVM ##########
Accuracy: 0.8395
Auc: 0.9222
Recall: 0.8936
Precision: 0.8073
F1: 0.8478
########## XGB ##########
Accuracy: 0.8463
Auc: 0.9203
Recall: 0.8769
Precision: 0.8271
F1: 0.8509
########## LightGBM ##########
Accuracy: 0.8515
Auc: 0.9263
Recall: 0.8855
Precision: 0.8295
F1: 0.8563 """

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_  # {'max_depth': None, 'max_features': 7, 'min_samples_split': 2, 'n_estimators': 500}

rf_best_grid.best_score_  # 0.8458

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8458
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9212
round(cv_results['test_recall'].mean(), 4)  # 0.8855
round(cv_results['test_precision'].mean(), 4)  # 0.8208
round(cv_results['test_f1'].mean(), 4)  # 0.8516

########## RF ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8470 --> 0.8458
# Auc: 0.9222 --> 0.9212
# Recall: 0.8791 --> 0.8855
# Precision: 0.8264 --> 0.8208
# F1: 0.8516 --> 0.8516


xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_best_grid.best_params_  # {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8594
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9319
round(cv_results['test_recall'].mean(), 4)  # 0.8963
round(cv_results['test_precision'].mean(), 4)  # 0.8346
round(cv_results['test_f1'].mean(), 4)  # 0.8643

# Xgboost modeli standar scaler olsa da aynı sonucu verdi ama RF daha iyileşti

########## XGBoost ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8463 ---> 0.8594
# Auc: 0.9203      ---> 0.9319
# Recall: 0.8769   ---> 0.8963
# Precision: 0.8271---> 0.8346
# F1: 0.8509       ---> 0.8643

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],  # 0.001
               "n_estimators": [100, 300, 500, 1000],  # 1000

               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_  # 'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 300}

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8571
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9273
round(cv_results['test_recall'].mean(), 4)  # 0.8981
round(cv_results['test_precision'].mean(), 4)  # 0.8300
round(cv_results['test_f1'].mean(), 4)  # 0.8626

########## LightGBM ########## Hyperparametre ile karşılaştırma
# Accuracy: 0.8515 -->0.8571
# Auc:      0.9263-->0.9273
# Recall:   0.8855-->0.8981
# Precision:0.8295-->0.8300
# F1:       0.8563-->0.8626
########## LightGBM ##########

################################################
# SVM
################################################

SVM_model = SVC(random_state=17)

SVM_params = {"C": [0.1, 1, 10],  # Ceza parametresi
              "kernel": ["linear", "rbf", "sigmoid"]}  # Çekirdek türü

SVM_best_grid = GridSearchCV(SVM_model, SVM_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

SVM_best_grid.best_params_  # {'C': 0.1, 'kernel': 'linear'}

SVM_final = SVM_model.set_params(**SVM_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(SVM_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
round(cv_results['test_accuracy'].mean(), 4)  # 0.8513
round(cv_results['test_roc_auc'].mean(), 4)  # 0.9252
round(cv_results['test_recall'].mean(), 4)  # 0.8855
round(cv_results['test_precision'].mean(), 4)  # 0.8291
round(cv_results['test_f1'].mean(), 4)  # 0.8561


########## SVM ########## Hyperparametre ile karşılaştırma
# Accuracy:  0.8395 --> 0.8499
# Auc:       0.9222--> 0.9248
# Recall:    0.8936 -->0.8809
# Precision: 0.8073--> 0.8297
# F1:        0.8478 -->0.8543
################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X, num=15, save=True)
plot_importance(xgboost_final, X, num=15, save=True)
plot_importance(lgbm_final, X, num=15, save=True)

random = X.sample(1, random_state=45)

# rf_final.predict(random)
# xgboost_final.predict(random)
# lgbm_final.predict(random)
# SVM_final.predict(random)

joblib.dump(rf_final, "rf_final.pkl")
joblib.dump(xgboost_final, "xgboost_final.pkl")
joblib.dump(lgbm_final, "lgbm_final.pkl")
joblib.dump(SVM_final, "SVM_final.pkl")

SVM_final_model_from_disc = joblib.load("voting_clf.pkl")
df = pd.read_csv("C:/Users/mehmet kupeli/PycharmProjects/pythonProject/datasets/data.csv", engine='python', sep=None)
random = X.sample(1, random_state=45)

SVM_final._model_from_disc.predict(random)

# BONUS
"""
# Başlangıç zamanını kaydet
start_time = time.time()

# Modeli kaydet
joblib.dump(SVM_final, "SVM_final.pkl")

# Bitiş zamanını kaydet
end_time = time.time()

# İşlem süresini hesapla
process_time = end_time - start_time
print(f"Model kaydetme süresi: {process_time} saniye") """
