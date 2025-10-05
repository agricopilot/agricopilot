from django.db import models
from django.utils import timezone

class CropDiagnosis(models.Model):
    """Model to store crop diagnosis data"""
    image = models.ImageField(upload_to='crop_images/', blank=True, null=True)
    symptoms = models.TextField()
    crop_type = models.CharField(max_length=100)
    diagnosis_result = models.JSONField(blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Crop Diagnosis - {self.crop_type} ({self.created_at.date()})"
