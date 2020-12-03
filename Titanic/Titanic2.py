
# coding: utf-8

# In[1]:


#导入pandas与numpy工具包
import pandas as pd
import numpy as np

#导入绘图工具包
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

#  %matplotlib inline  用在Jupyter notebook中具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。


# # 数据的初步分析

# In[2]:


# 分别从本地读取训练集和测试集数据
# 此处使用pandas，读取数据的同时转换为pandas独有的dataframe格式（二维数据表格）
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)
print(test.shape)


# In[3]:


train.head()


# In[4]:


train.describe()


# In[5]:


test.head()


# In[6]:


test.describe()


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


#查看训练集前5行数据
train.head()


# In[10]:


#查看训练集后5行数据
train.tail()


# In[11]:


# 查看训练集中生还和死亡人数
train['Survived'].value_counts()


# In[12]:


# 饼图显示生还/死亡乘客对比
train['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
# 保留2位小数,并且尽量使整个输出至少占用1个字符


# In[13]:


# 查看训练集舱位人数分布
train['Pclass'].value_counts()


# In[14]:


# 饼图显示训练集仓位人数分布
train['Pclass'].value_counts().plot.pie(autopct='%1.2f%%')


# In[15]:


# 查看测试集仓位人数分布
test['Pclass'].value_counts()


# In[16]:


# 饼图显示测试集舱位人数分布
train['Pclass'].value_counts().plot.pie(autopct='%1.2f%%')


# In[17]:


#采用seaborn绘图函数库作可视化分析
sns.countplot(x="Pclass", hue="Survived", data=train)


# In[18]:


# 比较舱位等级存活率
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(color=[['r','g','b']])


# In[19]:


#比较仓位等级中生存男女比例
train[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()


# In[20]:


# 查看不同舱位等级中男女生存数值
train.groupby(['Sex','Pclass','Survived'])['Survived'].count()


# In[21]:


# 查看训练集人员性别分布数值
train['Sex'].value_counts()


# In[22]:


# 饼图显示训练集人员性别分布
train['Sex'].value_counts().plot.pie(autopct='%1.2f%%')


# In[23]:


# 查看测试集人员性别分布数值
test['Sex'].value_counts()


# In[24]:


# 饼图显示测试集人员性别分布
test['Sex'].value_counts().plot.pie(autopct='%1.2f%%')


# In[25]:


# 查看性别与生存数值
train.groupby(['Sex','Survived'])['Survived'].count()


# In[26]:


# 柱状图显示人员性别与生存
sns.countplot(x="Sex", hue="Survived", data=train)## hue:指定分类变量


# In[27]:


# 比较性别与生存平均比率
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color= [['r','g']])


# In[28]:


# 查看乘客年龄分布
train['Age'].value_counts()


# In[29]:


# 查看总体的年龄分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train['Age'].hist(bins=70)
#bins指bin(箱子)的个数,即每张图柱子的个数 
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Age', showfliers=False)
plt.show()
# 盒状图中的中间线是median（中位数），与统计中mean（均值）29.699118有一定差异。
# showmeans	是否显示均值


# In[30]:


# 柱形图比对年龄-生存分布
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=50)
# map(function, iterable, …)  将function应用于iterable的每一个元素
plt.show()


# In[31]:


#小提琴图，比较年龄和生存关系
sns.violinplot(x='Survived',y='Age',data=train)


# In[32]:


# 换一种图形来对比一下
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
#aspect=4  纵横比，每个小图的横轴长度和纵轴的比
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


# In[33]:


#年龄特征分段
train['Age']=train['Age'].map(lambda x: 'child' if x<12 else 'youth' if x<30 else 'adult' if x<60 else 'old' if x<70 else 'too old' if x>=70 else 'null')

# 柱状图比较分段后的年龄-生存关系
sns.countplot(x="Age", hue="Survived", data=train)


# In[34]:


# 重新导入数据
train = pd.read_csv('E:\Competition\Kaggle_Titanic-master\Kaggle_Titanic-master/train.csv')

# 因为年龄有缺失值，暂时用200填充缺失值
train["Age"].fillna('200',inplace = True)
# inplace参数的取值：True、False, True：直接修改原对象, False：创建一个副本，修改副本，原对象不变（缺省默认）

# 按年龄划分的平均生存率
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
#if train["Age"].isnull() == True:
train["Age_int"] = train["Age"].astype(int)
average_age = train[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
# as_index:bool,默认为True 对于聚合输出,返回以组标签作为索引的对象。
sns.barplot(x='Age_int', y='Survived', data=average_age)
# 200岁那个就是缺失值人群的平均生存率。


# In[35]:


#再次导入原训练集
train = pd.read_csv('E:\Competition\Kaggle_Titanic-master\Kaggle_Titanic-master/train.csv')

# 小提琴图显示舱位-年龄-生存关系
f,ax=plt.subplots(1,2,figsize=(15,5))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
#split:将split设置为true则绘制分拆的violinplot以比较经过hue拆分后的两个量

# 小提琴显示年龄-性别-生存关系
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[36]:


# 查看训练集SibSp数据分布
train['SibSp'].value_counts()


# In[37]:


# 查看测试集SibSp数据分布
test['SibSp'].value_counts()


# In[38]:


# 查看数据分布柱状图
train['SibSp'].value_counts().plot.bar()


# In[39]:


# 按照比例查看
train['SibSp'].value_counts(sorted)


# In[40]:


# 按照比例显示柱状图
train['SibSp'].value_counts(sorted).plot.bar()


# In[41]:


# 查看不同SibSp人群下死亡/生存对比
sns.countplot(x="SibSp", hue="Survived", data=train)


# In[42]:


# 查看不同SibSp的生存比率
train[['SibSp','Survived']].groupby(['SibSp']).mean()


# In[43]:


# 查看不同SibSp的生存平均率
train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar().set_title('SibSp and Survived')


# In[44]:


# SibSp分段对比生存率
train['SibSp']=train['SibSp'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')

train['SibSp'].value_counts()

# 柱状图显示
sns.countplot(x="SibSp", hue="Survived", data=train)


# In[45]:


# 查看训练集Parch数据分布
train['Parch'].value_counts()


# In[46]:


# 查看测试集Parch数据分布
test['Parch'].value_counts()


# In[47]:


# 查看训练集数据分布柱状图
train['Parch'].value_counts().plot.bar()


# In[48]:


# 查看Parch与生存对比
sns.countplot(x="Parch", hue="Survived", data=train)


# In[49]:


# 查看不同Parch下的生存比率
train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar().set_title('Parch and Survived')


# In[50]:


# Parch分段对比生存率
train['Parch']=train['Parch'].map(lambda x: 'small' if x<1 else 'middle' if x<4 else 'large')

sns.countplot(x="Parch", hue="Survived", data=train)


# In[51]:


#再次导入原训练集
train = pd.read_csv('E:\Competition\Kaggle_Titanic-master\Kaggle_Titanic-master/train.csv')

# 绘图显示SibSp、Parch与乘客生存的对比
f, ax = plt.subplots(1, 2, figsize = (15, 5))
train[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax = ax[0])
ax[0].set_title('Parch and Survived')
train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')


# In[52]:


# 查看票号分布
train['Ticket'].value_counts()


# In[53]:


# 抽样看一下
train[train['Ticket'] == '1601']


# In[54]:


# 查看Fare分布
train['Fare'].value_counts()


# In[55]:


# 查看数据分布柱状图
train['Fare'].value_counts().plot.bar()


# In[56]:


# 查看总体的船票花费分布
plt.figure(figsize=(12, 5))
plt.subplot(121)
train['Fare'].hist(bins=70)
plt.xlabel('Fare')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Fare', showfliers=False)
plt.show()


# In[57]:


# 查看船票与舱位对比
train.groupby(['Fare', 'Pclass'])['Pclass'].count().plot.bar()


# In[58]:


# 小提琴图对比票价和生存关系
sns.violinplot(x = 'Survived', y = 'Fare', data=train)


# In[59]:


# 上图显示的很不均匀分布
#用numpy库里的对数函数对Fare的数值进行对数转换
train['Fare']=train['Fare'].map(lambda x : np.log(x + 1))
#作小提琴图：
sns.violinplot(x = 'Survived', y = 'Fare', data=train)


# In[60]:


#再次导入原训练集
train = pd.read_csv('E:\Competition\Kaggle_Titanic-master\Kaggle_Titanic-master/train.csv')

# 提取团体票的计数值，形成一个新列
train['Group_Ticket'] = train['Fare'].groupby(by = train['Ticket']).transform('count')

# 团体票除以票价计数值，求出每张票价格
train['Fare'] = train['Fare'] / train['Group_Ticket']

# 删除临时列
train.drop(['Group_Ticket'], axis=1, inplace=True)

# 查看数据分布柱状图
train['Fare'].value_counts().plot.bar()


# In[61]:


# 查看总体的船票花费分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train['Fare'].hist(bins=100)
plt.xlabel('Fare')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Fare', showfliers=False)
plt.show()


# In[62]:


#  Fare(团体票处理后数据)与生存对比
# 小提琴图对比票价和生存关系
sns.violinplot(x='Survived',y='Fare',data=train)


# In[63]:


# 上图显示的很不均匀分布
#用numpy库里的对数函数对Fare的数值进行对数转换
train['Fare']=train['Fare'].map(lambda x:np.log(x+1))
#作小提琴图：
sns.violinplot(x='Survived',y='Fare',data=train)


# In[64]:


#可以很明显的发现当log(Fare)在2、3处有明显变化。2的时候，死亡率是高于生存率的，而大于3的生存率是高于死亡率的，因此可以做如下分类：
train['Fare']=train['Fare'].map(lambda x: 'poor' if x < 2 else 'middle' if x < 3 else 'rich')

#作小提琴图：
sns.violinplot(x='Survived',y='Fare',data=train)


# #不同符号代表的港口含义
# #S = Southampton,南安普顿 （第1站）
# #C = Cherbourg, 瑟堡 （第2站）
# #Q = Queenstown, 皇后城 （第3站）

# In[65]:


# 查看数据分布
train['Embarked'].value_counts()


# In[66]:


# 查看数据分布柱状图
train['Embarked'].value_counts().plot.bar()


# In[67]:


# 查看数据分布饼图
train['Embarked'].value_counts().plot.pie(autopct='%1.2f%%')


# In[68]:


# Emabarked人群与生存对比
sns.countplot(x="Embarked", hue="Survived", data=train)


# In[69]:


# 不同登船港口-舱位对比
sns.countplot(x="Embarked", hue="Pclass", data=train)


# In[70]:


# 查看Cabin数据分布
train['Cabin'].value_counts()


# In[71]:


# 船舱编号-舱位对比
sns.countplot(x="Cabin", hue="Pclass", data=train)


# In[72]:


#有编号的为yes,没有的为no
train['Cabin']=train['Cabin'].map(lambda x:'yes' if type(x)==str else 'no')
#作图
sns.countplot(x="Cabin", hue="Survived", data=train)


# # 数据的处理阶段

# In[73]:


#导入pandas与numpy工具包
import pandas as pd
import numpy as np

#导入绘图包
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

#下面两句是解决绘图显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False   # 解决坐标轴负数的负号显示问题

import string
import warnings
warnings.filterwarnings('ignore')


# In[74]:


#分别从本地读取训练集和测试集数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[75]:


# 测试集中临时增加Survied列，并将该列数据填充为0
test['Survived'] = 0

# 把测试集数据追加到训练集后面，形成一个混合数据集。这里使用的是pandas的append方法
combined = train.append(test,sort=False)


# In[76]:


train.shape


# In[77]:


test.shape


# In[78]:


combined.shape


# In[79]:


combined.head()


# In[80]:


combined.tail()


# In[81]:


combined.info()


# In[82]:


# 数据集命名，为了方便后面的显示
train.name = 'Training Set' 
test.name = 'Test Set' 
combined.name = 'All Set'   

dfs = [train, test]


# In[83]:


# 定义缺失值显示函数
def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

# 调用函数，显示缺失值
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)


# In[84]:


# 先行处理的数据——Fare团体票处理

# 建立一个临时列，存放团体票的计数值
combined['Group_Ticket'] = combined['Fare'].groupby(by=combined['Ticket']).transform('count')

# 票价对应除以团体票计数值，得到每张票的真实价格。如果非团体票，那么就是除以1，价格不变
combined['Fare'] = combined['Fare'] / combined['Group_Ticket']

# 删除临时列
combined.drop(['Group_Ticket'], axis=1, inplace=True)


# # 数据相关性分析

# In[85]:


survived = train['Survived'].value_counts()[1]
not_survived = train['Survived'].value_counts()[0]
survived_per = survived / train.shape[0] * 100
not_survived_per = not_survived / train.shape[0] * 100

print('{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(survived, train.shape[0], survived_per))
print('{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(not_survived, train.shape[0], not_survived_per))

plt.figure(figsize=(6, 4))
sns.countplot(train['Survived'])

plt.xlabel('Survival', size=15, labelpad=15)
# # labelpad=15 设置轴名称离x轴的距离
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
# plt.xticks([0,1],[1,2],rotation=0)  [0,1]代表x坐标轴的0和1位置，[1,2]代表0,1位置的显示lable，rotation代表lable显示的旋转角度。
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)
# labelsize 参数labelsize用于设置刻度线标签的字体大小
plt.title('Training Set Survival Distribution', size=15, y=1.05)

plt.show() 


# In[86]:


# 训练集相关性
train_corr = train.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()##相关系数矩阵
# ascending=False代表降序
# df.stack() 列索引→行索引    df.unstack() 行索引→列索引
train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
train_corr.drop(train_corr.iloc[1::2].index, inplace=True)
train_corr_nd = train_corr.drop(train_corr[train_corr['Correlation Coefficient'] == 1.0].index)

# 训练集数据特征间的高相关性
corr = train_corr_nd['Correlation Coefficient'] > 0.1
train_corr_nd[corr]
# 这里是将一系列数据属性分为特征1和特征2，然后分析它们之间的相关性
# 舱位和票价之间的相关性最高


# In[87]:


# 测试集相关性
test_corr = test.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
# stack函数会将数据从”表格结构“变成”花括号结构“，即将其行索引变成列索引，反之，unstack函数将数据从”花括号结构“变成”表格结构“，
test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
test_corr.drop(test_corr.iloc[1::2].index, inplace=True)
test_corr_nd = test_corr.drop(test_corr[test_corr['Correlation Coefficient'] == 1.0].index)

# 测试集数据特征间的高相关性
corr = test_corr_nd['Correlation Coefficient'] > 0.1
test_corr_nd[corr]


# In[88]:


# 相关性显示图示
fig, axs = plt.subplots(nrows=2, figsize=(16, 16))

sns.heatmap(train.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})
sns.heatmap(test.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})
# annot: 默认为False，为True的话，会在格子上显示数字
# annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    # 参数axis的值为’x’、’y’、’both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’
    
axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()


# In[89]:


# 临时把测试集和训练集从混合集中分开，建两个分析用的临时数据集
train_data = combined[:891]
test_data = combined[891:]

# 删除空值
train_data.dropna(axis=0, how='any', inplace=True)

train_temp = train_data

# 删除空值
test_data.dropna(axis=0, how='any', inplace=True)

test_temp = test_data


# In[90]:


# 基于年龄和票价的生存特征分布
cont_features = ['Age', 'Fare']
surv = train_temp['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # 生存的特征分布
    sns.distplot(train_temp[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(train_temp[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])
    
    # 各个特征在数据集中的分布
    sns.distplot(train_temp[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(test_temp[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=10)
        axs[i][j].tick_params(axis='y', labelsize=10)
    
    axs[0][i].legend(loc='upper right', prop={'size': 10})
    axs[1][i].legend(loc='upper right', prop={'size': 10})
    axs[0][i].set_title('关于 {} 特征的生存分布'.format(feature), size=10, y=1.05)

axs[1][0].set_title('关于 {} 特征分布'.format('Age'), size=10, y=1.05)
axs[1][1].set_title('关于 {} 特征分布'.format('Fare'), size=10, y=1.05)
        
plt.show()


# In[91]:


# 基于其他几个特征的生存分布
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(10, 10))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=train_temp)
    
    plt.xlabel('{}'.format(feature), size=10, labelpad=15)
    plt.ylabel('乘客数量', size=10, labelpad=15)    
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 12})
    plt.title('在 {} 特征下的生存数量统计'.format(feature), size=8, y=1.05)

plt.show()


# # 特征数据处理

# In[92]:


# 取Embarked的众数（也就是数值最多的）
combined['Embarked'].mode()


# In[93]:


# 取Embarked的众数第一行
combined['Embarked'].mode().iloc[0]


# In[94]:


#总共缺失2个，采用众数填充
if combined['Embarked'].isnull().sum() != 0:
    combined['Embarked'].fillna(combined['Embarked'].mode().iloc[0], inplace=True)

combined.info()


# In[95]:


import re
# 在下面的代码中，我们通过正则提取了Title特征，正则表达式为(\w+\.)，它会在Name特征里匹配第一个以“.”号为结束的单词。同时，指定expand=False的参数会返回一个DataFrame。
# 西方姓名中间会加入称呼，比如小男童会在名字中间加入Master，女性根据年龄段及婚姻状况不同也会使用Miss 或 Mrs 等
# 这算是基于业务的理解做的衍生特征，原作者应该是考虑可以用作区分人的特征因此在此尝试清洗数据后加入

combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(combined['Title'], combined['Sex'])


# In[96]:


combined['Title'].value_counts()


# In[97]:


#将名称分类
combined['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer',inplace = True)
combined['Title'].replace(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'], 'Royalty',inplace =True )
combined['Title'].replace(['Mlle', 'Miss'], 'Miss',inplace =True )
combined['Title'].replace('Ms', 'Miss',inplace =True )
combined['Title'].replace(['Mme', 'Mrs'], 'Mrs',inplace =True )
combined['Title'].replace(['Mr'], 'Mr',inplace =True )
combined['Title'].replace(['Master'], 'Master',inplace =True )


# In[98]:


# 查看分类后的名称统计
combined['Title'].value_counts()


# In[99]:


# 下面我们印证一下包含Master的都是小童

# 使用一个临时数据表
temp = combined[combined['Title'].str.contains('Master')]
temp['Age'].value_counts()


# In[100]:


# 查看年龄分段后的生存率
combined[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[101]:


# Fare填充缺失值  注意，之前我们已经先处理了团体票，然后在这里才处理缺失值的
combined.info()


# In[102]:


combined['Fare'].isnull().sum()


# In[103]:


combined[combined['Fare'].isnull()]


# In[104]:


# 按一二三等舱各自的均价来对应填充NaN
if combined['Fare'].isnull().sum() != 0:
    combined['Fare'] = combined[['Fare']].fillna(combined.groupby('Pclass').transform('mean'))

# 查看填充后的数据
combined.iloc[1043]


# In[105]:


combined['Fare_Category'] = pd.qcut(combined['Fare'],13)
# q=13表示分成13个箱子

combined['Fare_Category']


# In[106]:


# 合并SibSp和Parch，得到家庭成员总数
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
combined[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[107]:


# 把家庭成员总数做分段处理
combined['FamilySizeCategory']=combined['FamilySize'].map(lambda x:'Single' if x<2 else 'small' if x<4 else 'middle' if x<8 else 'large')


# In[108]:


#求出Age为非空，同时Name中包含的Master的乘客年龄均值
ZZ = combined[combined['Age'].notnull() & combined['Title'].str.contains('Master')]['Age'].mean()

ZZ


# In[109]:


#使用这句无效
#combined[combined['Age'].isnull()  and combined['Title'].str.contains('Master')]['Age'].fillna(ZZ,inplace = True)

combined.loc[(combined['Title'] == 'Master') & (combined['Age'].isnull()), 'Age'] = ZZ
combined[combined['Age'].isnull()  & combined['Title'].str.contains('Master')]


# In[110]:


#combined.loc[65,'Age'] = ZZ
#combined.loc[159,'Age'] = ZZ
#combined.loc[176,'Age'] = ZZ
#combined.loc[709,'Age'] = ZZ
#combined.loc[244,'Age'] = ZZ
#combined.loc[339,'Age'] = ZZ
#combined.loc[344,'Age'] = ZZ
#combined.loc[417,'Age'] = ZZ

#combined


# # 使用混合预测模型预测Age
# 在这里使用了GradientBoostingRegressor、 RandomForestRegressor方法，本身可能也会引入预测误差，而且预测模型随着参数设定的不同，预测误差 会很大，未必就比简单里办法更好
# 但为什么要保留，主要还是基于学习的目的， 

# In[111]:


missing_age_df = pd.DataFrame(combined[['Pclass', 'Name', 'Sex', 'Age', 'FamilySize', 'FamilySizeCategory','Fare','Embarked', 'Title']])
missing_age_df = pd.get_dummies(missing_age_df,columns=[ 'Name', 'Sex','Embarked','FamilySizeCategory','Title'])

# 注意，这里没有对数值型数据做标准化处理
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]


# GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。

# In[112]:


# 下面先定义了一个训练和填充函数
from sklearn import ensemble
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.preprocessing import StandardScaler

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # 这里对训练数据做了标准化处理
    missing_age_X_train = StandardScaler().fit_transform(missing_age_X_train)
    missing_age_X_test = StandardScaler().fit_transform(missing_age_X_test)
    
    # GBM模型预测
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    #  cv=None 交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器
    # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    
    # 随机森林模型预测
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # 模型预测结果合并
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)
    # mode() 返回在某一数组或数据区域中出现频率最多的数值

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    #做了标准化以后，数据会变成np.array格式，这里再做一次转换
    missing_age_test = pd.DataFrame(missing_age_test)
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test


# In[113]:


combined.loc[(combined.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)


# In[114]:


# Age分段处理
combined['Age_group'] = combined['Age'].map(lambda x: 'child' if x<12 else 'youth' if x<18 else 'adult' if x<30 else 'middle' if x<50 else 'old' if x<70 else 'too old' if x>=70 else 'null')

by_age = combined.groupby('Age_group')['Survived'].mean()

by_age


# In[115]:


# Cabin处理

# 创建Deck列，根据Cabin列的第一个字母（M表示missing）
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
combined['Deck'] = combined['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

combined_decks = combined.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

#  transpose()函数的作用就是调换x,y,z的位置,也就是数组的索引值
#  map() 是一个Series的函数，DataFrame结构中没有map()。map()将一个自定义函数应用于Series结构中的每个元素(elements)。
#  apply()和applymap()是DataFrame结构中的函数，Series中没有。
#  它们的区别在于，apply()将一个函数作用于DataFrame中的每个行或者列，而applymap()是将函数做用于DataFrame中的所有元素(elements)。​

def get_pclass_dist(df):
    
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]
    
    # #当pclass不存在时，直接使用deck_counts[deck][pclass]，会报名为kayerror的错误。所以添加了报错处理。当报错时，先赋值为0。
    # #计算每个甲板上，不同阶级人数的占比
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count 
            except KeyError:
                deck_counts[deck][pclass] = 0
                
    df_decks = pd.DataFrame(deck_counts)    
    deck_percentages = {}

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]
        
    return deck_counts, deck_percentages

def display_pclass_dist(percentages):
    
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85
    
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
    plt.figure(figsize=(10, 5))
    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
    plt.show()    

all_deck_count, all_deck_per = get_pclass_dist(combined_decks)
display_pclass_dist(all_deck_per)


# In[116]:


# 把T甲板的乘客改到A甲板
# Passenger in the T deck is changed to A
idx = combined[combined['Deck'] == 'T'].index
combined.loc[idx, 'Deck'] = 'A'


# In[117]:


combined_decks_survived = combined.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

def get_survived_dist(df):
    
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]    

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
            
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]
        
    return surv_counts, surv_percentages

def display_surv_dist(percentages):
    
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))  
    bar_width = 0.85    

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]
    
    plt.figure(figsize=(10, 5))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.show()

all_surv_count, all_surv_per = get_survived_dist(combined_decks_survived)
display_surv_dist(all_surv_per)


# In[118]:


combined['Deck'] = combined['Deck'].replace(['A', 'B', 'C'], 'ABC')
combined['Deck'] = combined['Deck'].replace(['D', 'E'], 'DE')
combined['Deck'] = combined['Deck'].replace(['F', 'G'], 'FG')

combined['Deck'].value_counts()


# In[119]:


# 下面这段主要是统计Ticket的频率特征， 这个实际是考虑家庭团体票，号码是一样的

combined['Ticket_Frequency'] = combined.groupby('Ticket')['Ticket'].transform('count')

fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Frequency', hue='Survived', data=combined)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)

plt.show()


# In[120]:


# 下面这段主要是通过Surname把家庭进行归类

def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        # strip（) 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        
        for c in string.punctuation: # #如果字符是标点符号的话就将其替换为空格
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

combined['Family'] = extract_surname(combined['Name'])
# #关于string.punctuation，见这篇博客：https://blog.csdn.net/kongsuhongbaby/article/details/83181768
# 关于.split()：结果返回一个划分的列表
# 函数用来提取家族属性和头衔属性
# #关于split()，看这篇博客：https://blog.csdn.net/liuweiyuxiang/article/details/90936521


# In[121]:


train = combined.loc[:890]
test = combined[891:]
dfs = [train, test]


# In[122]:


# Creating a list of families and tickets that are occuring in both training and test set
# 下面这段是为了创建一个同时存在于训练集合测试集的家庭和Ticket列表

non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]
non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]
# unique(A) 返回与 A中相同的数据,但是不包含重复项

df_family_survival_rate = train.groupby('Family')['Survived', 'Family','FamilySize'].median()
df_ticket_survival_rate = train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()
# Median函数.功能:返回一组数值的中值

family_rates = {}
ticket_rates = {}
# df_family_survival_rate.index是指所有的家族名称

# 以下两段函数，分别生成了family_rates和ticket_rates两个字典。family_rates键代表家族名，值代表生存率的中位数。ticket_rates同理。
for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    # 检查家族这个属性是否在训练集和验证集中都出现过，并且成员数量>1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    # 检查票价这个属性是否在训练集和验证集中都出现过，并且成员数量>1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]


# In[123]:


# 这里考虑的是家庭生存率

mean_survival_rate = np.mean(train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

# 分为在训练集和测试集中都出现和只出现在df_train中这样两种。
# 利用上一段代码得到的family_rates和ticket_rates来创建属性。
for i in range(len(train)):
    if train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
train['Family_Survival_Rate'] = train_family_survival_rate
train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
test['Family_Survival_Rate'] = test_family_survival_rate
test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(train)):
    if train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(test)):
    if test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
train['Ticket_Survival_Rate'] = train_ticket_survival_rate
train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
test['Ticket_Survival_Rate'] = test_ticket_survival_rate
test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA


# In[124]:


# 把基于家庭计算的生存率和基于Tiket计算的生存率做个平均

for df in [train, test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2  


# In[125]:


train.info()


# In[126]:


test.info()


# In[127]:


train_data = train
test_data =test
train_data.to_csv('train_data.csv',index= False)
test_data.to_csv('test_data.csv',index= False)


# # 特征编码处理

# In[128]:


#导入pandas与numpy工具包
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

import string
import warnings
warnings.filterwarnings('ignore')


# In[129]:


train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')


# In[130]:


train.columns


# In[131]:


# 将训练数据分成标记和特征两部分

# 提取出训练集数据标记
y_train = train['Survived']

# 删除明确不需要的列
X_train = train.drop(['PassengerId', 'Survived','Name','Age','Ticket','Fare', 'Cabin','Family','Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA'],axis=1)

X_train.info()


# In[132]:


# 把PassengerId提取出来，后面用
Id = test['PassengerId']

# 删除明确不需要的列
X_test = test.drop(['PassengerId', 'Survived','Name','Age','Ticket','Fare', 'Cabin','Family','Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA'],axis=1)

X_test.info()


# In[133]:


# 对数值型的列做二值化分列处理，非数值的get_dummies会自动分列处理

# 对数值列Pclass做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Pclass, prefix= 'Pclass'))

# 对数值列SibSp做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.SibSp, prefix= 'SibSp'))

# 对数值列Parch做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Parch, prefix= 'Parch'))

# 因为测试集里面Parch多了一个数，训练集里面没有，如果不做补充，训练集和测试集维度会不一样
X_train['Parch_9'] = 0

# 对数值列FamilySize做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.FamilySize, prefix= 'FamilySize'))

# 对数值列Ticket_Frequency做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Ticket_Frequency, prefix= 'Ticket_Frequency'))

# 对数值列Survival_Rate做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Survival_Rate, prefix= 'Survival_Rate'))

# 对数值列Survival_Rate_NA做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Survival_Rate_NA, prefix= 'Survival_Rate_NA'))


# In[134]:


X_train['Parch'].value_counts()
# 注意这里的Parch，训练集和测试集的数值不完全相同，所以给训练集中补了一个数据。否则做完编码，训练集会少一列


# In[135]:


X_test['Parch'].value_counts()


# In[136]:


# 删除7个数值列  再次删除训练集中不必要的列
X_train = X_train.drop(['Pclass','SibSp','Parch','FamilySize','Ticket_Frequency','Survival_Rate','Survival_Rate_NA'],axis=1)


# In[137]:


X_train = pd.get_dummies(X_train)
encoded = list(X_train.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

X_train.info()


# In[138]:


# 测试集特征二值化编码

# 对数值列Pclass做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Pclass, prefix= 'Pclass'))

# 对数值列SibSp做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.SibSp, prefix= 'SibSp'))

# 对数值列Parch做二值化分列处理
X_test= X_test.join(pd.get_dummies(X_test.Parch, prefix= 'Parch'))

# 对数值列FamilySize做二值化分列处理
X_test =X_test.join(pd.get_dummies(X_test.FamilySize, prefix= 'FamilySize'))

# 对数值列Ticket_Frequency做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Ticket_Frequency, prefix= 'Ticket_Frequency'))

# 对数值列Survival_Rate做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Survival_Rate, prefix= 'Survival_Rate'))

# 对数值列Survival_Rate_NA做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Survival_Rate_NA, prefix= 'Survival_Rate_NA'))


# In[139]:


# 再次删除测试集中不需要的列

X_test = X_test.drop(['Pclass','SibSp','Parch','FamilySize','Ticket_Frequency','Survival_Rate','Survival_Rate_NA'],axis=1)


# In[140]:


X_test = pd.get_dummies(X_test)
encoded = list(X_test.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

X_test.info()


# 特征筛选
# 
# 特征筛选在这里的目的是通过几个机器学习模型，筛选出对结果影响最大的特征
# 
# 然后将最重要的特征合并起来为后面机器学习和预测使用

# In[141]:


from time import time
from sklearn import ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# In[142]:


# 定义特征筛选函数

def get_top_n_features(X_train, y_train, top_n_features):
    # 随机森林
    rf_est = RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)   
    #这里使用了网格搜索 GridSearchCV自动调最优参数
    rf_grid.fit(X_train,y_train)
    #将feature按Importance排序
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(X_train), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 25 Features from RF Classifier')
    print(str(features_top_n_rf[:25]))

    # AdaBoost
    ada_est = ensemble.AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(X_train, y_train)
    #排序
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(X_train),'importance': ada_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 25 Features from Ada Classifier')
    print(str(features_top_n_ada[:25]))

    # ExtraTree
    et_est = ensemble.ExtraTreesClassifier(random_state=42)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(X_train, y_train)
    #排序
    feature_imp_sorted_et = pd.DataFrame({'feature': list(X_train), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 25 Features from ET Classifier:')
    print(str(features_top_n_et[:25]))

    # 将三个模型挑选出来的前features_top_n_et合并
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et], ignore_index=True).drop_duplicates()

    return features_top_n


# In[143]:


# 特征筛选

# 这里选择了25个，也就是三种算法的前25个影响最大的特征值保留
# 处理完之后，合并为46个

feature_to_pick = 25
feature_top_n = get_top_n_features(X_train,y_train,feature_to_pick)
X_train = X_train[feature_top_n]
X_test = X_test[feature_top_n]


# In[144]:


X_train.to_csv('X_train.csv',index=False,sep=',')

X_test.to_csv('X_test.csv',index=False,sep=',')

y_train.to_csv('y_train.csv',index=False,sep=',')


# # 模型学习和预测

# In[145]:


#导入pandas与numpy工具包
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

import string
import warnings
warnings.filterwarnings('ignore')


# In[146]:


# 导入特征值数据

X_train = pd.read_csv('X_train.csv')

# 从原始数据中提取出Survived列，作为y_train。这里就是为了弄一个y_train出来
temp = pd.read_csv('train.csv')
y_train = temp['Survived']

X_test = pd.read_csv('X_test.csv')
# 注意，这里除了y_train导入的是前面经过特征工程编码后的数据


# In[147]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron


from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings('ignore')


# In[148]:


# 标准化编码  这里使用了StandardScaler编码，有的人是不用的

from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)

# 使用了StandardScaler编码，X_train和X_test就变成了np.array格式，后面如果要用pandas处理，还需要转换一下


# In[149]:


# 定义拟合曲线显示函数

from sklearn.model_selection import learning_curve
#import matplotlib.pyplot as plt

# 定义函数 plot_learning_curve 绘制学习曲线。train_sizes 初始化为 array([ 0.1  ,  0.325,  0.55 ,  0.775,  1\.   ]),cv 初始化为 10，以后调用函数时不再输入这两个变量

def plot_learning_curve(estimator, title, X_train, y_train, cv=10,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title) # 设置图的 title
    plt.xlabel('Training examples') # 横坐标
    plt.ylabel('Score') # 纵坐标
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=cv,
                                                            train_sizes=train_sizes) 
    train_scores_mean = np.mean(train_scores, axis=1) # 计算平均值
    train_scores_std = np.std(train_scores, axis=1) # 计算标准差
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid() # 设置背景的网格

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='g') # 设置颜色
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='g',
             label='traning score') # 绘制训练精度曲线
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r',
             label='testing score') # 绘制测试精度曲线
    plt.legend(loc='best')
    return plt

# 拟合曲线还是很重要的模型训练中的分析方法


# In[150]:


# 随机森林模型
# 这位大神的参数是经过反复测试的

leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1) 
leaderboard_model.fit(X_train,y_train)
g = plot_learning_curve(leaderboard_model, 'RFC', X_train,y_train) #调用定义的 plot_learning_curve 绘制学习曲线


# In[151]:


#  保存结果
pred1 = leaderboard_model.predict(X_test)
pred1 = pd.DataFrame(pred1)
pred1 = pred1.astype(int)
pred1['Survived'] = pred1

# 前面说了，因为格式的问题，这里还要再做一次转换
test = pd.read_csv('test.csv')

submission = pd.DataFrame({'PassengerId':test.loc[:,'PassengerId'],
                               'Survived':pred1.loc[:,'Survived']})
submission.to_csv('lead-1.csv',index=False,sep=',')

