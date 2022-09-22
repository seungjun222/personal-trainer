from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.


class User(AbstractUser):
    student_id = models.CharField(max_length=10)
    # profile_img = models.ImageField(null=True)
    # uploadedFile = models.ImageField(null=True)


class Document(models.Model):
    id=models.CharField(max_length=200,primary_key=True)
    title = models.CharField(max_length=200)
    uploadedFile = models.FileField(upload_to="Uploaded_Files/")
    trainerpath=models.CharField(max_length=200,blank=True)
    time=models.IntegerField(blank=True,null=True)
    dateTimeOfUpload = models.DateTimeField(auto_now=True)
