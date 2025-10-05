from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from datetime import datetime
from .models import CropDiagnosis
import requests

# Create your views here.

def main_view(request):
    """Render the main landing page with animation"""
    context = {
        'page_title': 'AgriCopilot - Smart Farming Assistant',
        'current_year': datetime.now().year,
    }
    return render(request, 'animation_load.html', context)

def page2_view(request):
    """Render the language selection page"""
    context = {
        'page_title': 'Choose Language - AgriCopilot',
        'supported_languages': ['English', 'Hausa', 'Yoruba', 'Igbo'],
    }
    return render(request, 'page2.html', context)

def auth_view(request):
    """Render the authentication page (login/signup)"""
    context = {
        'page_title': 'Login/Signup - AgriCopilot',
        'current_year': datetime.now().year,
    }
    return render(request, 'auth.html', context)

def page4_view(request):
    """Render the photo upload page for crop diagnosis"""
    context = {
        'page_title': 'Upload Crop Photo - AgriCopilot',
        'max_file_size': '10MB',
        'supported_formats': ['JPG', 'PNG', 'JPEG'],
    }
    return render(request, 'page4.html', context)

def page5_view(request):
    """Render the symptom description page"""
    context = {
        'page_title': 'Describe Symptoms - AgriCopilot',
        'common_crops': ['Maize', 'Rice', 'Cassava', 'Yam', 'Tomato', 'Pepper', 'Cowpea'],
        'symptom_categories': ['Leaf Issues', 'Stem Problems', 'Root Diseases', 'Fruit/Vegetable Issues'],
    }
    return render(request, 'page5.html', context)

def page6_view(request, diagnosis_id=None):
    """Render the AI analysis results page"""
    context = {
        'page_title': 'AI Analysis Results - AgriCopilot',
        'analysis_timestamp': datetime.now().isoformat(),
        'diagnosis_id': diagnosis_id,
    }
    return render(request, 'page6.html', context)

def homescreen_view(request):
    """Render the main dashboard/homescreen"""
    context = {
        'page_title': 'Dashboard - AgriCopilot',
        'current_date': datetime.now().strftime('%B %d, %Y'),
        'current_time': datetime.now().strftime('%H:%M'),
        'weather_data': {
            'temperature': '28°C',
            'condition': 'Sunny',
            'location': 'Your Farm'
        },
        'quick_actions': [
            {'name': 'Scan Crop', 'icon': 'camera', 'url': '/page4/'},
            {'name': 'Describe Issue', 'icon': 'edit', 'url': '/page5/'},
            {'name': 'View History', 'icon': 'history', 'url': '#'},
            {'name': 'Get Advice', 'icon': 'question-circle', 'url': '#'},
        ]
    }
    return render(request, 'homescreen.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def upload_crop_image(request):
    """Handle crop image upload and save to database"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)

        image_file = request.FILES['image']

        # Validate file type
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'error': 'Uploaded file must be an image'}, status=400)

        # Validate file size (max 10MB)
        if image_file.size > 10 * 1024 * 1024:
            return JsonResponse({'error': 'File size too large. Max 10MB allowed'}, status=400)

        # Create diagnosis record with image
        diagnosis = CropDiagnosis.objects.create(
            image=image_file,
            symptoms='',  # Will be updated in page5
            crop_type=''  # Will be updated in page5
        )

        return JsonResponse({
            'success': True,
            'diagnosis_id': diagnosis.id,
            'image_url': diagnosis.image.url,
            'message': 'Image uploaded successfully'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def save_symptoms(request):
    """Save symptoms description and crop type to database"""
    try:
        data = json.loads(request.body)
        diagnosis_id = data.get('diagnosis_id')
        symptoms = data.get('symptoms', '').strip()
        crop_type = data.get('crop_type', '').strip()

        if not diagnosis_id:
            return JsonResponse({'error': 'Diagnosis ID is required'}, status=400)

        if not symptoms:
            return JsonResponse({'error': 'Symptoms description is required'}, status=400)

        if not crop_type:
            return JsonResponse({'error': 'Crop type is required'}, status=400)

        # Update the diagnosis record
        try:
            diagnosis = CropDiagnosis.objects.get(id=diagnosis_id)
            diagnosis.symptoms = symptoms
            diagnosis.crop_type = crop_type
            diagnosis.save()

            return JsonResponse({
                'success': True,
                'diagnosis_id': diagnosis.id,
                'message': 'Symptoms saved successfully'
            })

        except CropDiagnosis.DoesNotExist:
            return JsonResponse({'error': 'Diagnosis record not found'}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_diagnosis_data(request, diagnosis_id):
    """Get diagnosis data for page6 display"""
    try:
        diagnosis = CropDiagnosis.objects.get(id=diagnosis_id)
        return JsonResponse({
            'diagnosis_id': diagnosis.id,
            'image_url': diagnosis.image.url if diagnosis.image else None,
            'symptoms': diagnosis.symptoms,
            'crop_type': diagnosis.crop_type,
            'diagnosis_result': diagnosis.diagnosis_result,
            'confidence_score': diagnosis.confidence_score,
            'created_at': diagnosis.created_at.isoformat()
        })
    except CropDiagnosis.DoesNotExist:
        return JsonResponse({'error': 'Diagnosis not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def run_crop_analysis(request, diagnosis_id):
    """Run crop analysis using the stored image and symptoms"""
    try:
        diagnosis = CropDiagnosis.objects.get(id=diagnosis_id)

        if not diagnosis.image:
            return JsonResponse({'error': 'No image found for analysis'}, status=400)

        if not diagnosis.symptoms:
            return JsonResponse({'error': 'No symptoms found for analysis'}, status=400)

        # Prepare symptoms description for API
        symptoms_description = f"{diagnosis.symptoms} (Crop: {diagnosis.crop_type})"

        # Create FormData for the crop doctor API call
        import requests
        import mimetypes

        # Prepare the API call to crop-doctor endpoint
        api_url = request.build_absolute_uri('/crop-doctor/')
        headers = {
            'Authorization': 'Bearer agricopilot404',
            'symptoms': symptoms_description
        }

        # Get file path and content type
        file_path = diagnosis.image.path
        content_type = mimetypes.guess_type(file_path)[0] or 'image/jpeg'

        # Open the file and create multipart form data
        with open(file_path, 'rb') as image_file:
            files = {'image': (diagnosis.image.name, image_file, content_type)}

            # Make the API call
            response = requests.post(api_url, headers=headers, files=files)

        if response.status_code != 200:
            return JsonResponse({'error': f'API call failed: {response.status_code}'}, status=500)

        diagnosis_result = response.json()

        # Save the diagnosis result
        diagnosis.diagnosis_result = diagnosis_result
        diagnosis.confidence_score = diagnosis_result.get('confidence_score', 0)
        diagnosis.save()

        return JsonResponse(diagnosis_result)

    except CropDiagnosis.DoesNotExist:
        return JsonResponse({'error': 'Diagnosis not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def multilingual_chat(request):
    """Handle multilingual chat queries for agricultural advice"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header != 'Bearer agricopilot404':
            return JsonResponse({'error': 'Unauthorized'}, status=401)

        # Parse JSON data
        data = json.loads(request.body)
        query = data.get('query', '')

        if not query:
            return JsonResponse({'error': 'Query parameter is required'}, status=400)

        # Simulate multilingual response (replace with actual AI/ML logic)
        response = generate_multilingual_response(query)

        return JsonResponse({
            'query': query,
            'response': response,
            'language': detect_language(query),
            'timestamp': '2025-10-04T12:00:00Z'
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def disaster_summarizer(request):
    """Summarize disaster reports for agricultural impact assessment"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header != 'Bearer agricopilot404':
            return JsonResponse({'error': 'Unauthorized'}, status=401)

        # Parse JSON data
        data = json.loads(request.body)
        report = data.get('report', '')

        if not report:
            return JsonResponse({'error': 'Report parameter is required'}, status=400)

        # Simulate disaster summarization (replace with actual AI/ML logic)
        summary = generate_disaster_summary(report)

        return JsonResponse({
            'original_report': report,
            'summary': summary,
            'severity_level': assess_severity(report),
            'affected_area': extract_location(report),
            'recommendations': generate_recommendations(summary),
            'timestamp': '2025-10-04T12:00:00Z'
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def crop_doctor(request):
    """Analyze crop diseases from symptoms and images"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header != 'Bearer agricopilot404':
            return JsonResponse({'error': 'Unauthorized'}, status=401)

        # Get symptoms from headers (as per API spec)
        symptoms = request.headers.get('symptoms', '')

        # Get uploaded image
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Image file is required'}, status=400)

        image_file = request.FILES['image']

        # Validate file type
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'error': 'Uploaded file must be an image'}, status=400)

        # Simulate crop disease analysis (replace with actual AI/ML logic)
        diagnosis = analyze_crop_disease(symptoms, image_file)

        return JsonResponse({
            'symptoms': symptoms,
            'diagnosis': diagnosis,
            'confidence_score': 0.85,  # Simulated confidence
            'treatment_recommendations': generate_treatment_plan(diagnosis),
            'preventive_measures': generate_preventive_measures(diagnosis),
            'timestamp': '2025-10-04T12:00:00Z'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Helper functions for AI/ML simulation (replace with actual implementations)

def generate_multilingual_response(query):
    """Simulate multilingual chat response"""
    # This would integrate with a multilingual LLM
    responses = {
        'rust': 'Leaf rust can be prevented by planting resistant varieties, proper spacing, and fungicide application.',
        'pest': 'For pest control, use integrated pest management with beneficial insects and organic pesticides.',
        'water': 'Proper irrigation scheduling prevents water stress. Use drip irrigation for efficiency.',
        'fertilizer': 'Soil testing helps determine fertilizer needs. Use balanced NPK ratios for optimal growth.'
    }

    query_lower = query.lower()
    for key, response in responses.items():
        if key in query_lower:
            return response

    return "Thank you for your agricultural question. Our experts will provide personalized advice based on your specific situation."

def detect_language(query):
    """Simple language detection (replace with actual language detection)"""
    # Basic detection - in real implementation, use a proper language detection library
    if any(word in query.lower() for word in ['hello', 'how', 'what', 'prevent', 'maize']):
        return 'en'
    return 'unknown'

def generate_disaster_summary(report):
    """Simulate disaster report summarization"""
    # This would use NLP to summarize disaster reports
    return f"Summary: {report[:100]}... (Impact assessment: High priority agricultural disaster response needed)"

def assess_severity(report):
    """Assess disaster severity"""
    severity_keywords = ['severe', 'destroyed', 'displaced', 'devastated', 'catastrophic']
    if any(keyword in report.lower() for keyword in severity_keywords):
        return 'high'
    return 'medium'

def extract_location(report):
    """Extract location from disaster report"""
    # Simple location extraction - in real implementation, use geocoding/NLP
    locations = ['Kano', 'Lagos', 'Abuja', 'Port Harcourt']
    for location in locations:
        if location.lower() in report.lower():
            return location
    return 'Unknown'

def generate_recommendations(summary):
    """Generate disaster response recommendations"""
    return [
        "Immediate assessment of affected farmlands",
        "Distribution of emergency seeds and fertilizers",
        "Coordination with local agricultural extension services",
        "Implementation of quick recovery programs"
    ]

def analyze_crop_disease(symptoms, image_file):
    """Simulate crop disease analysis"""
    # This would use computer vision and ML models
    symptoms_lower = symptoms.lower()

    # Common crop disease patterns
    if 'rot' in symptoms_lower:
        if 'maize' in symptoms_lower or 'corn' in symptoms_lower:
            return {
                'disease': 'Maize Stalk Rot',
                'affected_crop': 'Maize',
                'causative_agent': 'Fusarium or Pythium fungi',
                'description': 'Fungal infection causing rotting of stalks and roots'
            }
        elif 'cassava' in symptoms_lower:
            return {
                'disease': 'Cassava Root Rot',
                'affected_crop': 'Cassava',
                'causative_agent': 'Fusarium or Phytophthora fungi',
                'description': 'Fungal disease causing root decay and plant death'
            }
        else:
            return {
                'disease': 'Root Rot',
                'affected_crop': 'Various Crops',
                'causative_agent': 'Soil-borne fungi',
                'description': 'Fungal infection causing root system decay'
            }

    elif 'die' in symptoms_lower or 'death' in symptoms_lower or 'dying' in symptoms_lower:
        return {
            'disease': 'Plant Death Syndrome',
            'affected_crop': 'Various Crops',
            'causative_agent': 'Multiple pathogens or environmental stress',
            'description': 'Sudden or progressive plant death from various causes'
        }

    elif 'yellow leaves' in symptoms_lower and 'black spots' in symptoms_lower:
        return {
            'disease': 'Late Blight',
            'affected_crop': 'Tomato',
            'causative_agent': 'Phytophthora infestans',
            'description': 'Fungal disease causing dark lesions on leaves and fruits'
        }
    elif 'yellow' in symptoms_lower:
        return {
            'disease': 'Nutrient Deficiency',
            'affected_crop': 'Various Crops',
            'causative_agent': 'Nitrogen or Magnesium deficiency',
            'description': 'Yellowing due to insufficient nutrients'
        }

    elif 'spots' in symptoms_lower or 'lesions' in symptoms_lower:
        return {
            'disease': 'Leaf Spot Disease',
            'affected_crop': 'Various Crops',
            'causative_agent': 'Various fungal pathogens',
            'description': 'Fungal infection causing spots and lesions on leaves'
        }

    elif 'wilt' in symptoms_lower:
        return {
            'disease': 'Fusarium Wilt',
            'affected_crop': 'Various Crops',
            'causative_agent': 'Fusarium oxysporum',
            'description': 'Vascular disease causing wilting and yellowing'
        }

    else:
        return {
            'disease': 'Unknown',
            'affected_crop': 'Unidentified',
            'causative_agent': 'To be determined',
            'description': 'Further analysis required'
        }

def generate_treatment_plan(diagnosis):
    """Generate treatment recommendations"""
    disease = diagnosis.get('disease', '')

    if disease == 'Late Blight':
        return [
            "Apply copper-based fungicide immediately",
            "Remove and destroy affected plant parts",
            "Improve air circulation between plants",
            "Avoid overhead watering"
        ]
    elif 'Deficiency' in disease:
        return [
            "Conduct soil test to identify specific deficiency",
            "Apply appropriate fertilizers (NPK)",
            "Adjust pH if necessary",
            "Use foliar sprays for quick correction"
        ]
    elif 'Rot' in disease:
        return [
            "Improve soil drainage to prevent waterlogging",
            "Apply fungicide treatments (e.g., metalaxyl)",
            "Remove and destroy infected plants",
            "Practice crop rotation with non-host crops",
            "Use disease-resistant varieties"
        ]
    elif 'Death Syndrome' in disease:
        return [
            "Identify and address underlying cause (disease, pests, or stress)",
            "Remove dead plants immediately",
            "Test soil for pathogens and nutrient levels",
            "Implement integrated pest management",
            "Improve irrigation and drainage practices"
        ]
    elif 'Spot Disease' in disease:
        return [
            "Apply appropriate fungicides (copper-based)",
            "Remove infected leaves",
            "Improve air circulation",
            "Avoid overhead irrigation",
            "Use disease-resistant varieties"
        ]
    elif 'Wilt' in disease:
        return [
            "Use resistant varieties",
            "Practice long crop rotations (4-5 years)",
            "Soil fumigation if severe",
            "Avoid planting susceptible crops",
            "Improve soil health with organic matter"
        ]
    else:
        return ["Consult local agricultural extension service for diagnosis"]

def generate_preventive_measures(diagnosis):
    """Generate preventive measures"""
    return [
        "Plant disease-resistant varieties",
        "Practice crop rotation",
        "Maintain proper plant spacing",
        "Regular field monitoring and scouting",
        "Proper irrigation management"
    ]