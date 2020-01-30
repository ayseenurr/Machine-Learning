
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[14]:


data=pd.read_csv('eksikveriler.csv') 


# In[11]:


data.info() # veriler hakkında bilgi edinmek için kullanılır.Burada yas sütununda 2 tane eksik veri olduğunu görüyoruz.


# Eksik verileri bulmak ve onlar üzerinde işlem yapmak için birden fazla yöntem bulunmaktadır.Bende burada elimden geldiği kadar göstermeye çalışacağım. 

# In[17]:


missing_values_count = data.isnull().sum() #Tüm kolonlardaki eksik veri sayısını bulmak için pandas kütüphanesinde bulunan isnull fonksiyonunu kullanıyoruz.

missing_values_count


# In[19]:


total_missing = missing_values_count.sum() # toplam kaçtane eksik veri olduğunu buluyoruz
total_missing


# In[30]:


# eksik veri bulunan sütunu silmek için pandas kütüphanesine ait olan dropna fonksiyonunu kullanıyoruz.
new_data=data.dropna(axis=1)
new_data


# In[32]:


new_data=data.dropna() # Burada eksik veri olan satırları sildik.
new_data


# In[34]:


# Bir diğer yöntem ise pandas kütüphanesinde bulunan fillna fonksiyonu ile eksik verileri istediğimiz değer ile doldurabiliriz.
new_data=data.fillna(0) 
new_data


# In[42]:


#Burada örnek olması bakımından her sütunun ortalama değerini hesaplayıp eksik olan veriler yerine fillna fonksiyonu ile doldurduk.
mean=data.mean()  
new_data=data.fillna(mean) 
new_data


# In[41]:


# Bir başka yöntem ise replace fonksiyonu burada ise yukarıda olduğu gibi ortalama değeri hesaplayıp eksik veriler ile yer değiştirme işlemi yapılıyor.

mean=data.mean() 
new_data=data.replace(np.NaN,mean)
new_data



# In[51]:


# Ve son olarak Imputation işlemi yapıyoruz. Sklearn kütüphanesini kullanacağız.

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) # ortalama hesaplamak içim


new_data=data.iloc[:,1:4].values # hangi kolonların alıncağını seçiyoruz

imputer=imputer.fit(new_data)
new_data=imputer.transform(new_data)

new_data

