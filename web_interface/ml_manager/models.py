from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class TrainingJob(models.Model):
    # Foreign Key to built-in User
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Store ML parameters (e.g., {"epochs": 10, "lr": 0.001})
    parameters = models.JSONField(default=dict, blank=True)
    
    # Workflow status
    is_approved = models.BooleanField(default=False)
    status = models.CharField(
        max_length=20, 
        choices=[('pending', 'Pending'), ('running', 'Running'), ('done', 'Done')],
        default='pending'
    )

    created_at = models.DateTimeField(auto_now_add=True)

