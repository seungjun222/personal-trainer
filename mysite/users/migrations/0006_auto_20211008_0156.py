# Generated by Django 3.2.6 on 2021-10-07 16:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0005_auto_20211001_0000'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='trainerpath',
            field=models.CharField(blank=True, max_length=200),
        ),
        migrations.AlterField(
            model_name='document',
            name='uploadedFile',
            field=models.FileField(upload_to='Uploaded_Files/'),
        ),
    ]
