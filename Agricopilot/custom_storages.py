from django.conf import settings
from storages.backends.azure_storage import AzureStorage


class AzureMediaStorage(AzureStorage):
    """
    Custom storage class for media files (user uploads)
    Uses a separate container from static files
    """
    account_name = settings.AZURE_ACCOUNT_NAME
    account_key = settings.AZURE_ACCOUNT_KEY
    azure_container = settings.AZURE_MEDIA_CONTAINER
    expiration_secs = None


class AzureStaticStorage(AzureStorage):
    """
    Custom storage class for static files (CSS, JS, app images)
    Uses a separate container from media files
    """
    account_name = settings.AZURE_ACCOUNT_NAME
    account_key = settings.AZURE_ACCOUNT_KEY
    azure_container = settings.AZURE_STATIC_CONTAINER
    expiration_secs = None