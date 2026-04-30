from django.contrib import admin
from .models import TrainingJob
from .services import execute_ml_workflow

# Register your models here.
@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "is_approved", 'status', 'created_at')

    list_editable = ('is_approved',)
    actions = ("approve_and_run_training",)

    def save_model(self, request, obj, form, change):
        was_approved = False
        if change and obj.pk:
            was_approved = (
                TrainingJob.objects.filter(pk=obj.pk, is_approved=True).exists()
            )

        super().save_model(request, obj, form, change)

        if obj.is_approved and not was_approved:
            execute_ml_workflow(obj)

    @admin.action(description="Approve and Start ML Training")
    def approve_and_run_training(self, request, queryset):
        for job in queryset:
            if not job.is_approved:
                job.is_approved = True
                job.save()
                # Trigger the ML Service
                execute_ml_workflow(job)

        self.message_user(request, "Training jobs started on the shared network.")
