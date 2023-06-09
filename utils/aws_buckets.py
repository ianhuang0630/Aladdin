import os
from typing import Type
import boto3
from boto3.s3.transfer import TransferConfig
from loguru import logger

# resource setup for s3 buckets
with open('credentials/aws_access_key_id', 'r') as f:
    aws_access_key_id = f.readline().strip()
with open('credentials/aws_secret_access_key', 'r') as f:
    aws_secret_access_key = f.readline().strip()
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                aws_secret_access_key=aws_secret_access_key)
# utils.aws_buckets.s3 is usable now from other modules
                
# defaults
transfer_config = TransferConfig(max_concurrency=4)
upload_extra_args = {
    'ACL': 'public-read'
} # everything uploaded should be


def get_available_buckets(s3_object=None) -> set:
    if s3_object is None:
        s3_object = s3 # default
    return set([el['Name'] for el in s3_object.list_buckets()['Buckets']])

available_buckets = get_available_buckets()




def upload_file(bucket_name:str, local_path:str, remote_name:str,
                override_transfer_config:Type[TransferConfig] = None,
                override_extra_args:dict = None,
                update_extra_args:dict = None
                ) -> str:
    assert os.path.exists(local_path), f"local path {local_path} does not exist"
    global available_buckets
    
    # check if the bucket_name is a valid bucket.
    if bucket_name not in available_buckets:
        # make the bucket, and then update available buckets
        s3.create_bucket(Bucket=bucket_name)        
        available_buckets = get_available_buckets()
        assert bucket_name in available_buckets, f'creating the bucket "{bucket_name}" failed'
        logger.info(f'Created the bucket {bucket_name}')


    if override_transfer_config is None:
        override_transfer_config = transfer_config  # using default global
    if override_extra_args is None:    
        override_extra_args = upload_extra_args
    if update_extra_args is not None:
        override_extra_args.update(update_extra_args)

    # NOTE may raise exceptions
    s3.upload_file(local_path, bucket_name, remote_name, 
                   Config=override_transfer_config,
                   ExtraArgs=override_extra_args)

    # preparing download url. 
    bucket_location = s3.get_bucket_location(Bucket=bucket_name) 
    region_code = bucket_location['LocationConstraint']
    link = f"https://{bucket_name}.s3.{region_code}.amazonaws.com/{remote_name}"
    return link


if __name__=='__main__':
    url = upload_file('aladdin-outputs', 'README.md', 'test.md')
    print(url)
    pass
