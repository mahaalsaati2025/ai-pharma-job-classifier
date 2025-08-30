#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
استخدام النموذج المدرب لتصنيف الوظائف
Job Classification Prediction Script

استخدام:
python predict_job.py
"""

import pickle
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# تحميل بيانات NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("تحذير: قد تحتاج لتحميل بيانات NLTK يدوياً")

def load_model():
    """تحميل النموذج المدرب والمعالجات"""
    try:
        # تحميل النموذج
        with open("best_classifier.pkl", "rb") as file:
            classifier = pickle.load(file)
        
        # تحميل معالج النصوص
        with open("features_extractor.pkl", "rb") as file:
            features_extractor = pickle.load(file)
        
        # تحميل تطابق الفئات (إن وجد)
        try:
            with open("category_mapping.pkl", "rb") as file:
                category_mapping = pickle.load(file)
        except:
            category_mapping = None
        
        print("✅ تم تحميل النموذج بنجاح")
        return classifier, features_extractor, category_mapping
    
    except FileNotFoundError as e:
        print(f"❌ لم يتم العثور على ملف النموذج: {e}")
        print("يرجى تشغيل Jobs_Classifiers_Fixed.py أولاً لتدريب النموذج")
        return None, None, None
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        return None, None, None

# دوال معالجة النص (نفس الدوال من الكود الأصلي)
def remove_tags(text):
    """إزالة HTML tags"""
    if not text:
        return ""
    remove = re.compile(r'<.*?>')
    return re.sub(remove, '', str(text))

def special_char(text):
    """إزالة الأحرف الخاصة"""
    if not text:
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
    if not text:
        return ""
    return str(text).lower()

def remove_stopwords(text):
    """إزالة كلمات الوقف"""
    if not text:
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

def preprocess_text(text):
    """معالجة النص بنفس الطريقة المستخدمة في التدريب"""
    if not text:
        return ""
    
    text = remove_tags(text)
    text = special_char(text)
    text = convert_lower(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text

def predict_job_category(job_description, classifier, features_extractor):
    """تصنيف وصف الوظيفة"""
    try:
        # معالجة النص
        processed_text = preprocess_text(job_description)
        
        if not processed_text:
            return "غير محدد", 0
        
        # تحويل إلى features
        text_features = features_extractor.transform([processed_text])
        
        # التنبؤ
        prediction = classifier.predict(text_features)
        
        # الحصول على احتماليات التصنيف
        try:
            probabilities = classifier.predict_proba(text_features)
            confidence = max(probabilities[0]) * 100
        except:
            confidence = 0
        
        return prediction[0], confidence
    
    except Exception as e:
        print(f"خطأ في التصنيف: {e}")
        return "غير محدد", 0

def display_categories():
    """عرض الفئات المتاحة"""
    categories = [
        "1. Pharmaceutical, Healthcare and Medical Sales",
        "2. Clinical Research",
        "3. Pharmaceutical Marketing", 
        "4. Manufacturing & Operations",
        "5. Science",
        "6. Medical Affairs / Pharmaceutical Physician",
        "7. Regulatory Affairs",
        "8. Medical Information and Pharmacovigilance",
        "9. Data Management and Statistics",
        "10. Quality-assurance",
        "11. Pharmacy"
    ]
    
    print("\n📋 فئات الوظائف المتاحة:")
    print("=" * 50)
    for category in categories:
        print(f"   {category}")
    print("=" * 50)

def run_examples(classifier, features_extractor):
    """تشغيل أمثلة تجريبية"""
    test_jobs = [
        "responsible for clinical trials and patient recruitment in pharmaceutical research",
        "marketing manager for pharmaceutical products and medical devices", 
        "quality assurance specialist ensuring compliance with FDA regulations",
        "sales representative for medical equipment and pharmaceutical products",
        "data analyst responsible for statistical analysis of clinical trial results",
        "regulatory affairs manager preparing drug submissions to health authorities",
        "pharmacist dispensing medications and providing patient counseling",
        "manufacturing engineer optimizing pharmaceutical production processes"
    ]
    
    print("\n=== أمثلة تجريبية ===")
    print("=" * 60)
    
    for i, job in enumerate(test_jobs, 1):
        category, confidence = predict_job_category(job, classifier, features_extractor)
        print(f"\n{i}. الوصف:")
        print(f"   {job}")
        print(f"   🎯 التصنيف: {category}")
        print(f"   📊 الثقة: {confidence:.1f}%")
        print("-" * 60)

def interactive_mode(classifier, features_extractor):
    """الوضع التفاعلي"""
    print("\n=== الوضع التفاعلي ===")
    print("💡 نصائح:")
    print("   - أدخل وصف الوظيفة باللغة الإنجليزية")
    print("   - كن محدداً في الوصف للحصول على نتائج أفضل")
    print("   - اكتب 'exit' أو 'quit' للخروج")
    print("   - اكتب 'examples' لرؤية أمثلة")
    print("   - اكتب 'categories' لرؤية الفئات المتاحة")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 أدخل وصف الوظيفة: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'خروج', 'q']:
                print("👋 شكراً لاستخدام مصنف الوظائف!")
                break
            
            elif user_input.lower() in ['examples', 'أمثلة']:
                run_examples(classifier, features_extractor)
                continue
            
            elif user_input.lower() in ['categories', 'فئات']:
                display_categories()
                continue
            
            elif user_input.lower() in ['help', 'مساعدة']:
                print("\n📚 الأوامر المتاحة:")
                print("   - examples: عرض أمثلة تجريبية")
                print("   - categories: عرض الفئات المتاحة")
                print("   - help: عرض هذه المساعدة")
                print("   - exit/quit: الخروج من البرنامج")
                continue
            
            elif not user_input:
                print("⚠️  يرجى إدخال وصف للوظيفة")
                continue
            
            # تصنيف الوظيفة
            category, confidence = predict_job_category(user_input, classifier, features_extractor)
            
            print(f"\n🎯 النتيجة:")
            print(f"   التصنيف المتوقع: {category}")
            print(f"   مستوى الثقة: {confidence:.1f}%")
            
            # تقييم الثقة
            if confidence >= 90:
                print("   ✅ ثقة عالية جدا