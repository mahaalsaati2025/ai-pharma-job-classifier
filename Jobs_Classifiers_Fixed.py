# Author: Maha Shakir - Fixed Version
# Fixed by: Claude AI Assistant

# Import libraries/packages
import pickle
import re
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("تحذير: قد تحتاج لتحميل بيانات NLTK يدوياً")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Configuration
show_plots = True
palette = sns.color_palette("Blues", 11)  # Updated for 11 categories

# Create sample dataset if original doesn't exist
def create_sample_dataset():
    """إنشاء بيانات تجريبية في حالة عدم وجود الملف الأصلي"""
    
    categories = [
        'Pharmaceutical, Healthcare and Medical Sales',
        'Clinical Research', 
        'Pharmaceutical Marketing',
        'Manufacturing & Operations',
        'Science',
        'Medical Affairs / Pharmaceutical Physician',
        'Regulatory Affairs',
        'Medical Information and Pharmacovigilance', 
        'Data Management and Statistics',
        'Quality-assurance',
        'Pharmacy'
    ]
    
    sample_descriptions = [
        # Pharmaceutical Sales
        "pharmaceutical sales representative responsible for promoting medical products to healthcare professionals",
        "medical device sales specialist working with hospitals and clinics",
        "healthcare sales manager overseeing regional pharmaceutical product distribution",
        
        # Clinical Research
        "clinical research coordinator managing patient recruitment and trial protocols", 
        "clinical data manager responsible for collecting and analyzing trial results",
        "research scientist conducting pharmaceutical clinical studies",
        
        # Pharmaceutical Marketing
        "pharmaceutical marketing manager developing promotional strategies for new drugs",
        "medical marketing specialist creating educational materials for healthcare providers",
        "brand manager for pharmaceutical products and market analysis",
        
        # Manufacturing & Operations  
        "manufacturing engineer optimizing pharmaceutical production processes",
        "operations manager overseeing drug manufacturing facilities",
        "production supervisor ensuring quality manufacturing standards",
        
        # Science
        "research scientist developing new pharmaceutical compounds", 
        "laboratory technician conducting chemical analysis and testing",
        "biochemist studying drug interactions and molecular mechanisms",
        
        # Medical Affairs
        "medical affairs specialist providing scientific support for pharmaceutical products",
        "pharmaceutical physician reviewing clinical data and medical literature", 
        "medical science liaison communicating with key opinion leaders",
        
        # Regulatory Affairs
        "regulatory affairs specialist ensuring FDA compliance for drug approvals",
        "regulatory manager preparing submissions for health authorities",
        "compliance officer managing pharmaceutical regulations and guidelines",
        
        # Medical Information  
        "medical information specialist responding to healthcare provider inquiries",
        "pharmacovigilance associate monitoring drug safety and adverse events",
        "drug safety officer analyzing safety data and risk assessments",
        
        # Data Management
        "biostatistician analyzing clinical trial statistical data",
        "data manager ensuring quality and integrity of clinical databases", 
        "statistical programmer developing analysis plans for studies",
        
        # Quality Assurance
        "quality assurance specialist ensuring GMP compliance in manufacturing",
        "QA manager overseeing quality control processes and audits",
        "quality control analyst testing pharmaceutical products for specifications",
        
        # Pharmacy
        "clinical pharmacist providing medication therapy management",
        "hospital pharmacist dispensing medications and counseling patients",
        "pharmaceutical care specialist optimizing drug therapy outcomes"
    ]
    
    # Create balanced dataset
    data = []
    descriptions_per_category = len(sample_descriptions) // len(categories)
    
    for i, category in enumerate(categories):
        start_idx = i * descriptions_per_category
        end_idx = start_idx + descriptions_per_category
        
        for desc in sample_descriptions[start_idx:end_idx]:
            data.append({'job_description': desc, 'category': category})
        
        # Add extra descriptions if available
        if i < len(sample_descriptions) % len(categories):
            extra_idx = len(categories) * descriptions_per_category + i
            if extra_idx < len(sample_descriptions):
                data.append({'job_description': sample_descriptions[extra_idx], 'category': category})
    
    df = pd.DataFrame(data)
    df.to_csv('dataset0.csv', index=False)
    print(f"✅ تم إنشاء ملف بيانات تجريبي بـ {len(df)} عينة")
    return df

# Load the dataset
try:
    if not os.path.exists('dataset0.csv'):
        print("⚠️  ملف البيانات غير موجود، سيتم إنشاء بيانات تجريبية...")
        dataset = create_sample_dataset()
    else:
        dataset = pd.read_csv("dataset0.csv", on_bad_lines='skip')
        print("✅ تم تحميل ملف البيانات بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل البيانات: {e}")
    print("سيتم إنشاء بيانات تجريبية...")
    dataset = create_sample_dataset()

# Display dataset information
print("\n=== معلومات البيانات ===")
print(f"حجم البيانات: {dataset.shape}")
print(f"الأعمدة: {list(dataset.columns)}")
print(f"عدد الفئات: {dataset['category'].nunique()}")

print("\n=== عينة من البيانات ===")
print(dataset.head())

print("\n=== توزيع الفئات ===")
print(dataset['category'].value_counts())

# Create category mapping
target_category = dataset['category'].unique()
dataset['categoryId'] = dataset['category'].factorize()[0]

category_df = dataset[['category', 'categoryId']].drop_duplicates().sort_values('categoryId')
category_to_id = dict(category_df.values)
id_to_category = dict(category_df[['categoryId', 'category']].values)

print("\n=== تطابق الفئات ===")
for cat_id, cat_name in id_to_category.items():
    print(f"{cat_id}: {cat_name}")

# Visualization
if show_plots:
    plt.figure(figsize=(15, 8))
    counts = dataset['category'].value_counts()
    colors = sns.color_palette("Blues", len(counts))
    
    ax = counts.plot(kind="bar", color=colors, figsize=(15, 8))
    plt.title("توزيع فئات الوظائف", fontsize=16, pad=20)
    plt.xlabel("فئة الوظيفة", fontsize=12)
    plt.ylabel("عدد الوظائف", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Pie chart
    plt.figure(figsize=(12, 10))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=palette[:len(counts)], startangle=45)
    plt.title("نسب فئات الوظائف", fontsize=16, pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('category_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

# NLP Processing Functions
def remove_tags(text):
    """إزالة HTML tags"""
    if pd.isna(text):
        return ""
    remove = re.compile(r'<.*?>')
    return re.sub(remove, '', str(text))

def special_char(text):
    """إزالة الأحرف الخاصة"""
    if pd.isna(text):
        return ""
    result = ''
    for char in str(text):
        if char.isalnum():
            result += char
        else:
            result += ' '
    return result

def convert_lower(text):
    """تحويل إلى أحرف صغيرة"""
    if pd.isna(text):
        return ""
    return str(text).lower()

def remove_stopwords(text):
    """إزالة كلمات الوقف"""
    if pd.isna(text):
        return []
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(str(text))
        return [word for word in words if word not in stop_words and len(word) > 2]
    except:
        return str(text).split()

def lemmatize_word(words):
    """استخراج جذور الكلمات"""
    if not words:
        return ""
    try:
        wordnet = WordNetLemmatizer()
        if isinstance(words, list):
            return " ".join([wordnet.lemmatize(word) for word in words])
        else:
            return str(words)
    except:
        if isinstance(words, list):
            return " ".join(words)
        return str(words)

# Apply preprocessing
print("\n=== معالجة النصوص ===")
print("1. إزالة HTML tags...")
dataset['job_description'] = dataset['job_description'].apply(remove_tags)

print("2. إزالة الأحرف الخاصة...")
dataset['job_description'] = dataset['job_description'].apply(special_char)

print("3. تحويل لأحرف صغيرة...")
dataset['job_description'] = dataset['job_description'].apply(convert_lower)

print("4. إزالة كلمات الوقف...")
dataset['job_description'] = dataset['job_description'].apply(remove_stopwords)

print("5. استخراج جذور الكلمات...")
dataset['job_description'] = dataset['job_description'].apply(lemmatize_word)

print("✅ تمت معالجة النصوص بنجاح")

# Feature extraction
print("\n=== استخراج المميزات ===")
x = dataset['job_description']
y = dataset['category']

# TF-IDF Vectorization
max_features = min(5000, len(dataset) * 2)  # Adjust for small datasets
features_extractor = TfidfVectorizer(
    max_features=max_features,
    min_df=1,
    max_df=0.9,
    ngram_range=(1, 2)
)

X_features = features_extractor.fit_transform(x).toarray()
print(f"شكل المميزات: {X_features.shape}")

# Train-test split
test_size = min(0.3, 0.5)  # Adjust for small datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=test_size, random_state=42, 
    shuffle=True, stratify=y
)

print(f"بيانات التدريب: {X_train.shape[0]} عينة")
print(f"بيانات الاختبار: {X_test.shape[0]} عينة")

# Model training and evaluation
performance_list = []

def run_model(model_name):
    """تدريب وتقييم النموذج"""
    print(f"\n🔄 تدريب نموذج: {model_name}")
    
    try:
        # Model selection
        if model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'Multinomial Naive Bayes':
            model = MultinomialNB(alpha=1.0)
        elif model_name == 'Support Vector Classifier':
            model = SVC(probability=True, random_state=42)
        elif model_name == 'Decision Tree':
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'K Nearest Neighbors':
            model = KNeighborsClassifier(n_neighbors=min(5, len(y_train)//2))
        elif model_name == 'Gaussian Naive Bayes':
            model = GaussianNB()
        
        # Train model
        classifier = OneVsRestClassifier(model)
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
        
        # Evaluation
        for data, pred, data_name in [(y_train, y_pred_train, 'Training'), 
                                      (y_test, y_pred_test, 'Testing')]:
            accuracy = accuracy_score(data, pred) * 100
            precision, recall, f1score, _ = score(data, pred, average='weighted')
            
            print(f"  {data_name} - دقة: {accuracy:.2f}%")
            print(f"  {data_name} - دقة: {precision:.3f}, استرجاع: {recall:.3f}, F1: {f1score:.3f}")
            
            # Confusion matrix
            if show_plots and data_name == 'Testing':
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(data, pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=classifier.classes_,
                           yticklabels=classifier.classes_)
                plt.title(f'مصفوفة الخلط - {model_name}')
                plt.ylabel('الفئة الحقيقية')
                plt.xlabel('الفئة المتوقعة')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
            
            # Store performance
            performance_list.append({
                'Model': model_name,
                'Data': data_name,
                'Accuracy': round(accuracy, 2),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'F1': round(f1score, 3)
            })
        
        return classifier
    
    except Exception as e:
        print(f"❌ خطأ في تدريب {model_name}: {e}")
        return None

# Train all models
print("\n=== تدريب النماذج ===")
models = [
    'Logistic Regression',
    'Random Forest', 
    'Multinomial Naive Bayes',
    'Decision Tree',
    'K Nearest Neighbors'
]

trained_models = {}
for model_name in models:
    model = run_model(model_name)
    if model is not None:
        trained_models[model_name] = model

# Performance comparison
print("\n=== مقارنة الأداء ===")
if performance_list:
    performance_df = pd.DataFrame(performance_list)
    test_performance = performance_df[performance_df['Data'] == 'Testing']
    
    print("\nأداء النماذج على بيانات الاختبار:")
    print(test_performance[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']].to_string(index=False))
    
    # Find best model
    best_model_name = test_performance.loc[test_performance['Accuracy'].idxmax(), 'Model']
    best_accuracy = test_performance['Accuracy'].max()
    
    print(f"\n🏆 أفضل نموذج: {best_model_name} بدقة {best_accuracy}%")
    
    # Save best model
    if best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        
        try:
            with open("best_classifier.pkl", "wb") as f:
                pickle.dump(best_model, f)
            
            with open("features_extractor.pkl", "wb") as f:
                pickle.dump(features_extractor, f)
            
            with open("category_mapping.pkl", "wb") as f:
                pickle.dump(id_to_category, f)
            
            print("✅ تم حفظ النموذج الأفضل والمعالجات")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النموذج: {e}")

# Interactive testing
def predict_job_category(job_description, model, vectorizer):
    """تصنيف وصف وظيفة جديد"""
    try:
        # Preprocess
        processed = remove_tags(job_description)
        processed = special_char(processed)
        processed = convert_lower(processed)
        processed = remove_stopwords(processed)
        processed = lemmatize_word(processed)
        
        # Transform and predict
        features = vectorizer.transform([processed])
        prediction = model.predict(features)[0]
        
        # Get confidence if possible
        try:
            proba = model.predict_proba(features)
            confidence = max(proba[0]) * 100
        except:
            confidence = 0
        
        return prediction, confidence
    
    except Exception as e:
        print(f"خطأ في التصنيف: {e}")
        return "غير محدد", 0

# Test with examples
print("\n=== اختبار النموذج ===")
if trained_models and best_model_name in trained_models:
    test_examples = [
        "clinical research coordinator managing patient trials",
        "pharmaceutical sales representative promoting medical devices",
        "quality assurance specialist ensuring manufacturing compliance",
        "marketing manager for pharmaceutical products"
    ]
    
    best_model = trained_models[best_model_name]
    
    for i, example in enumerate(test_examples, 1):
        category, confidence = predict_job_category(example, best_model, features_extractor)
        print(f"{i}. الوصف: {example}")
        print(f"   التصنيف: {category} (ثقة: {confidence:.1f}%)")
        print()

# Interactive loop
print("\n=== الاختبار التفاعلي ===")
if trained_models and best_model_name in trained_models:
    while True:
        user_input = input("\nأدخل وصف الوظيفة (أو 'exit' للخروج): ")
        if user_input.lower() in ['exit', 'quit', 'خروج']:
            break
        
        if user_input.strip():
            category, confidence = predict_job_category(user_input, best_model, features_extractor)
            print(f"التصنيف المتوقع: {category}")
            print(f"مستوى الثقة: {confidence:.1f}%")

print("\n✅ اكتمل تشغيل المشروع بنجاح!")
print("📁 الملفات المحفوظة:")
print("   - best_classifier.pkl (النموذج الأفضل)")  
print("   - features_extractor.pkl (معالج النصوص)")
print("   - category_mapping.pkl (تطابق الفئات)")
print("   - الرسوم البيانية (.png files)")