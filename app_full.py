#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تطبيق تصنيف الوظائف الكامل مع الذكاء الاصطناعي
"""

from flask import Flask, request, render_template_string, jsonify
import re
import pickle
import os

# محاولة استيراد المكتبات
try:
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    ML_AVAILABLE = True
    print("✅ جميع مكتبات الذكاء الاصطناعي متوفرة")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️  بعض مكتبات ML غير متوفرة: {e}")

app = Flask(__name__)

# تحميل بيانات NLTK
if ML_AVAILABLE:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True) 
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ تم تحميل بيانات NLTK")
    except:
        print("⚠️  تحذير: مشكلة في تحميل بيانات NLTK")

# متغيرات النموذج
classifier = None
features_extractor = None

def create_sample_data():
    """إنشاء بيانات تجريبية وتدريب نموذج بسيط"""
    if not ML_AVAILABLE:
        return False
    
    print("🤖 إنشاء وتدريب نموذج الذكاء الاصطناعي...")
    
    # بيانات تجريبية
    sample_data = [
        ("clinical research coordinator managing patient trials and regulatory compliance", "Clinical Research"),
        ("pharmaceutical sales representative promoting medical devices to hospitals", "Pharmaceutical, Healthcare and Medical Sales"),
        ("quality assurance specialist ensuring GMP compliance in drug manufacturing", "Quality-assurance"),
        ("regulatory affairs manager preparing FDA submissions for new drug approvals", "Regulatory Affairs"),
        ("marketing manager developing promotional strategies for pharmaceutical products", "Pharmaceutical Marketing"),
        ("pharmacist dispensing medications and providing patient counseling services", "Pharmacy"),
        ("manufacturing engineer optimizing pharmaceutical production processes", "Manufacturing & Operations"),
        ("research scientist developing new pharmaceutical compounds", "Science"),
        ("medical affairs specialist providing scientific support for pharmaceutical products", "Medical Affairs / Pharmaceutical Physician"),
        ("pharmacovigilance associate monitoring drug safety and adverse events", "Medical Information and Pharmacovigilance"),
        ("biostatistician analyzing clinical trial statistical data", "Data Management and Statistics"),
    ]
    
    # إنشاء DataFrame
    df = pd.DataFrame(sample_data, columns=['job_description', 'category'])
    
    # معالجة بسيطة للنص
    def simple_preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    df['processed'] = df['job_description'].apply(simple_preprocess)
    
    # استخراج المميزات
    global features_extractor, classifier
    features_extractor = TfidfVectorizer(max_features=1000, stop_words='english')
    X = features_extractor.fit_transform(df['processed'])
    y = df['category']
    
    # تدريب النموذج
    base_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier = OneVsRestClassifier(base_classifier)
    classifier.fit(X, y)
    
    print("✅ تم تدريب النموذج بنجاح")
    return True

def predict_job_category(job_description):
    """تصنيف وصف الوظيفة"""
    if not ML_AVAILABLE or classifier is None:
        # تصنيف بسيط بناءً على الكلمات المفتاحية
        return simple_keyword_classification(job_description)
    
    try:
        # معالجة النص
        processed_text = job_description.lower()
        processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
        
        # التنبؤ
        features = features_extractor.transform([processed_text])
        prediction = classifier.predict(features)[0]
        
        # حساب الثقة
        try:
            probabilities = classifier.predict_proba(features)
            confidence = max(probabilities[0]) * 100
        except:
            confidence = 85.0
        
        return prediction, confidence
    
    except Exception as e:
        print(f"خطأ في التصنيف: {e}")
        return simple_keyword_classification(job_description)

def simple_keyword_classification(job_description):
    """تصنيف بسيط بناءً على الكلمات المفتاحية"""
    text = job_description.lower()
    
    keywords = {
        'Clinical Research': ['clinical', 'research', 'trial', 'study', 'patient', 'protocol'],
        'Pharmaceutical, Healthcare and Medical Sales': ['sales', 'representative', 'promote', 'sell', 'market', 'customer'],
        'Quality-assurance': ['quality', 'assurance', 'compliance', 'gmp', 'audit', 'control'],
        'Regulatory Affairs': ['regulatory', 'affairs', 'fda', 'submission', 'approval', 'regulation'],
        'Pharmaceutical Marketing': ['marketing', 'brand', 'promotional', 'advertising', 'campaign'],
        'Pharmacy': ['pharmacist', 'pharmacy', 'dispensing', 'medication', 'counseling'],
        'Manufacturing & Operations': ['manufacturing', 'production', 'operations', 'facility', 'process'],
        'Science': ['scientist', 'research', 'laboratory', 'analysis', 'development'],
        'Medical Affairs / Pharmaceutical Physician': ['medical', 'affairs', 'physician', 'clinical', 'scientific'],
        'Medical Information and Pharmacovigilance': ['pharmacovigilance', 'safety', 'adverse', 'monitoring'],
        'Data Management and Statistics': ['data', 'statistics', 'analysis', 'biostatistician', 'database']
    }
    
    scores = {}
    for category, words in keywords.items():
        score = sum(1 for word in words if word in text)
        if score > 0:
            scores[category] = score
    
    if scores:
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(90, 70 + scores[best_category] * 5)
        return best_category, confidence
    
    return 'Science', 75.0

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مصنف الوظائف الصيدلانية والطبية - الذكاء الاصطناعي</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .ai-status {
            background: {{ ai_color }};
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }
        .content { padding: 40px; }
        .input-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 2px dashed #e9ecef;
            transition: all 0.3s ease;
        }
        .input-section:hover {
            border-color: #3498db;
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .input-section h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 20px;
            margin-left: 10px;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
        }
        .btn.clear {
            background: #95a5a6;
            box-shadow: 0 5px 15px rgba(149, 165, 166, 0.3);
        }
        .result-section {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #27ae60;
            margin-top: 30px;
            {% if not result %}display: none;{% endif %}
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-title {
            color: #27ae60;
            font-size: 1.5em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .category-result {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .confidence-bar {
            background: #ecf0f1;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-fill {
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            height: 100%;
            transition: width 1s ease;
            border-radius: 4px;
        }
        .examples-section {
            margin-top: 40px;
            background: #fff5f5;
            padding: 30px;
            border-radius: 15px;
            border: 2px solid #ffe6e6;
        }
        .examples-section h3 {
            color: #e74c3c;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        .example {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid #e74c3c;
        }
        .example:hover {
            transform: translateX(10px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .categories-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        .category-item {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
            transition: transform 0.3s ease;
        }
        .category-item:hover { transform: translateY(-5px); }
        @media (max-width: 768px) {
            .container { margin: 10px; }
            .header { padding: 30px 20px; }
            .header h1 { font-size: 2em; }
            .content { padding: 20px; }
            .categories-list { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 مصنف الوظائف الصيدلانية والطبية</h1>
            <p>تقنية الذكاء الاصطناعي لتصنيف الوظائف باستخدام معالجة اللغة الطبيعية</p>
        </div>

        <div class="ai-status">
            {{ ai_status }}
        </div>

        <div class="content">
            <div class="input-section">
                <h3>📝 أدخل وصف الوظيفة</h3>
                <form method="POST">
                    <textarea 
                        name="job_description" 
                        placeholder="مثال: clinical research coordinator managing patient trials and regulatory compliance..."
                        required
                    >{{ job_description or '' }}</textarea>
                    <br>
                    <button type="submit" class="btn">🤖 تصنيف بالذكاء الاصطناعي</button>
                    <button type="button" class="btn clear" onclick="clearForm()">🗑️ مسح</button>
                </form>
            </div>

            {% if result %}
            <div class="result-section">
                <div class="result-title">
                    <span>🎯</span>
                    <span>نتيجة التصنيف</span>
                </div>
                <div class="category-result">
                    <h4>{{ result[0] }}</h4>
                    <p>مستوى الثقة: {{ "%.1f"|format(result[1]) }}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {{ result[1] }}%;"></div>
                    </div>
                    {% if result[1] >= 90 %}
                        <p style="color: #27ae60; margin-top: 10px;">✅ ثقة عالية جداً</p>
                    {% elif result[1] >= 80 %}
                        <p style="color: #f39c12; margin-top: 10px;">🟡 ثقة جيدة</p>
                    {% elif result[1] >= 70 %}
                        <p style="color: #e67e22; margin-top: 10px;">🟠 ثقة متوسطة</p>
                    {% else %}
                        <p style="color: #e74c3c; margin-top: 10px;">🔴 ثقة منخفضة</p>
                    {% endif %}
                </div>
            </div>
            {% endif %}

            <div class="examples-section">
                <h3>💡 أمثلة للاختبار</h3>
                <div class="example" onclick="fillExample(this)">
                    clinical research coordinator managing patient trials and regulatory compliance
                </div>
                <div class="example" onclick="fillExample(this)">
                    pharmaceutical sales representative promoting medical devices to hospitals
                </div>
                <div class="example" onclick="fillExample(this)">
                    quality assurance specialist ensuring GMP compliance in drug manufacturing
                </div>
                <div class="example" onclick="fillExample(this)">
                    regulatory affairs manager preparing FDA submissions for new drug approvals
                </div>
                <div class="example" onclick="fillExample(this)">
                    marketing manager developing promotional strategies for pharmaceutical products
                </div>
                <div class="example" onclick="fillExample(this)">
                    pharmacist dispensing medications and providing patient counseling services
                </div>
            </div>

            <div style="margin-top: 40px;">
                <h3 style="color: #2c3e50; margin-bottom: 20px;">📋 فئات الوظائف المتاحة</h3>
                <div class="categories-list">
                    <div class="category-item">المبيعات الصيدلانية والطبية</div>
                    <div class="category-item">البحوث السريرية</div>
                    <div class="category-item">التسويق الصيدلاني</div>
                    <div class="category-item">التصنيع والعمليات</div>
                    <div class="category-item">العلوم</div>
                    <div class="category-item">الشؤون الطبية</div>
                    <div class="category-item">الشؤون التنظيمية</div>
                    <div class="category-item">المعلومات الطبية ومراقبة الأدوية</div>
                    <div class="category-item">إدارة البيانات والإحصاء</div>
                    <div class="category-item">ضمان الجودة</div>
                    <div class="category-item">الصيدلة</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function fillExample(element) {
            document.querySelector('textarea[name="job_description"]').value = element.textContent.trim();
        }

        function clearForm() {
            document.querySelector('textarea[name="job_description"]').value = '';
        }

        console.log('🤖 تطبيق الذكاء الاصطناعي لتصنيف الوظائف جاهز!');
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    job_description = ''
    
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        
        if job_description:
            result = predict_job_category(job_description)
    
    # تحديد حالة الذكاء الاصطناعي
    if ML_AVAILABLE and classifier is not None:
        ai_status = "🤖 الذكاء الاصطناعي نشط - نموذج Random Forest مدرب"
        ai_color = "#27ae60"
    elif ML_AVAILABLE:
        ai_status = "🔄 الذكاء الاصطناعي قيد التحميل - تصنيف بالكلمات المفتاحية"
        ai_color = "#f39c12"
    else:
        ai_status = "⚠️ وضع التصنيف البسيط - بعض المكتبات غير متوفرة"
        ai_color = "#e67e22"
    
    return render_template_string(HTML_TEMPLATE, 
                                result=result, 
                                job_description=job_description,
                                ai_status=ai_status,
                                ai_color=ai_color)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    try:
        data = request.get_json()
        if not data or 'job_description' not in data:
            return jsonify({'error': 'job_description مطلوب', 'success': False}), 400
        
        job_description = data['job_description'].strip()
        if not job_description:
            return jsonify({'error': 'وصف الوظيفة فارغ', 'success': False}), 400
        
        category, confidence = predict_job_category(job_description)
        
        return jsonify({
            'success': True,
            'category': category,
            'confidence': round(confidence, 2),
            'input': job_description,
            'ai_model': 'active' if (ML_AVAILABLE and classifier is not None) else 'simple'
        })
    
    except Exception as e:
        return jsonify({'error': f'حدث خطأ: {str(e)}', 'success': False}), 500

if __name__ == '__main__':
    print("🚀 بدء تشغيل تطبيق تصنيف الوظائف مع الذكاء الاصطناعي")
    print("=" * 60)
    
    # تدريب النموذج إذا كانت المكتبات متوفرة
    if ML_AVAILABLE:
        create_sample_data()
    
    print("🌐 الرابط: http://localhost:5000")
    print("📊 API: http://localhost:5000/api/classify")
    print("=" * 60)
    print("💡 استخدم Ctrl+C لإيقاف الخادم")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")
    except Exception as e:
        print(f"❌ خطأ: {e}")