#!/usr/bin/env python3
"""
Deployment script for decline prediction API
Handles Docker containerization and AWS deployment
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
from datetime import datetime
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/deployment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_command(command, description, logger, check=True):
    """Run shell command and log output"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=check
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        if result.stderr and result.returncode == 0:
            logger.warning(f"Warnings: {result.stderr}")
        
        logger.info(f"Completed: {description}")
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def create_dockerfile(config, logger):
    """Create Dockerfile for the application"""
    logger.info("Creating Dockerfile...")
    
    dockerfile_content = f"""
# Decline Prediction API Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE {config['api']['port']}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config['api']['port']}/health || exit 1

# Run the application
CMD ["python", "api/decline_api.py"]
"""
    
    with open('../Dockerfile', 'w') as f:
        f.write(dockerfile_content.strip())
    
    logger.info("Dockerfile created successfully")
    return True

def create_docker_compose(config, logger):
    """Create docker-compose.yml for local development"""
    logger.info("Creating docker-compose.yml...")
    
    docker_compose_content = f"""
version: '3.8'

services:
  decline-api:
    build: .
    ports:
      - "{config['api']['port']}:{config['api']['port']}"
    environment:
      - ENVIRONMENT=development
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config['api']['port']}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - decline-api

volumes:
  redis_data:
"""
    
    with open('../docker-compose.yml', 'w') as f:
        f.write(docker_compose_content.strip())
    
    logger.info("docker-compose.yml created successfully")
    return True

def create_requirements_file(logger):
    """Create requirements.txt file"""
    logger.info("Creating requirements.txt...")
    
    requirements = """
torch==2.0.1
dgl==1.1.2
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
redis==5.0.1
pydantic==2.5.0
pyyaml==6.0.1
matplotlib==3.7.2
seaborn==0.12.2
python-multipart==0.0.6
"""
    
    with open('../requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    logger.info("requirements.txt created successfully")
    return True

def create_nginx_config(config, logger):
    """Create nginx configuration for load balancing"""
    logger.info("Creating nginx.conf...")
    
    nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream decline_api {{
        server decline-api:{config['api']['port']};
    }}

    server {{
        listen 80;
        
        location / {{
            proxy_pass http://decline_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }}
        
        location /health {{
            proxy_pass http://decline_api/health;
            access_log off;
        }}
    }}
}}
"""
    
    with open('../nginx.conf', 'w') as f:
        f.write(nginx_config.strip())
    
    logger.info("nginx.conf created successfully")
    return True

def create_aws_deployment_files(config, logger):
    """Create AWS deployment files"""
    logger.info("Creating AWS deployment files...")
    
    # ECS Task Definition
    task_definition = {
        "family": "decline-prediction",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "1024",
        "memory": "2048",
        "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
        "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
        "containerDefinitions": [
            {
                "name": "decline-api",
                "image": "ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/decline-prediction:latest",
                "portMappings": [
                    {
                        "containerPort": config['api']['port'],
                        "protocol": "tcp"
                    }
                ],
                "environment": [
                    {"name": "ENVIRONMENT", "value": "production"}
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/decline-prediction",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                },
                "healthCheck": {
                    "command": ["CMD-SHELL", f"curl -f http://localhost:{config['api']['port']}/health || exit 1"],
                    "interval": 30,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 60
                }
            }
        ]
    }
    
    with open('../aws/ecs-task-definition.json', 'w') as f:
        json.dump(task_definition, f, indent=2)
    
    # ECS Service Definition
    service_definition = {
        "serviceName": "decline-prediction",
        "cluster": "decline-prediction-cluster",
        "taskDefinition": "decline-prediction",
        "desiredCount": 2,
        "launchType": "FARGATE",
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": ["subnet-12345", "subnet-67890"],
                "securityGroups": ["sg-12345"],
                "assignPublicIp": "ENABLED"
            }
        },
        "loadBalancers": [
            {
                "targetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/decline-prediction/12345",
                "containerName": "decline-api",
                "containerPort": config['api']['port']
            }
        ],
        "healthCheckGracePeriodSeconds": 60
    }
    
    with open('../aws/ecs-service-definition.json', 'w') as f:
        json.dump(service_definition, f, indent=2)
    
    # CloudFormation template
    cloudformation_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Decline Prediction API Infrastructure",
        "Resources": {
            "VPC": {
                "Type": "AWS::EC2::VPC",
                "Properties": {
                    "CidrBlock": "10.0.0.0/16",
                    "EnableDnsHostnames": True,
                    "EnableDnsSupport": True
                }
            },
            "ECSCluster": {
                "Type": "AWS::ECS::Cluster",
                "Properties": {
                    "ClusterName": "decline-prediction-cluster"
                }
            },
            "ApplicationLoadBalancer": {
                "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
                "Properties": {
                    "Name": "decline-prediction-alb",
                    "Scheme": "internet-facing",
                    "Type": "application"
                }
            }
        }
    }
    
    with open('../aws/cloudformation-template.json', 'w') as f:
        json.dump(cloudformation_template, f, indent=2)
    
    logger.info("AWS deployment files created successfully")
    return True

def build_docker_image(config, logger):
    """Build Docker image"""
    logger.info("Building Docker image...")
    
    image_name = f"{config['deployment']['docker']['image_name']}:{config['deployment']['docker']['tag']}"
    
    success = run_command(
        f"docker build -t {image_name} ../",
        f"Build Docker image {image_name}",
        logger
    )
    
    if success:
        logger.info(f"Docker image {image_name} built successfully")
    
    return success

def run_local_deployment(config, logger):
    """Run local deployment with docker-compose"""
    logger.info("Starting local deployment...")
    
    # Stop any existing containers
    run_command(
        "docker-compose -f ../docker-compose.yml down",
        "Stop existing containers",
        logger,
        check=False
    )
    
    # Start services
    success = run_command(
        "docker-compose -f ../docker-compose.yml up -d",
        "Start services with docker-compose",
        logger
    )
    
    if success:
        logger.info(f"Local deployment started on port {config['api']['port']}")
        logger.info("Services:")
        logger.info(f"  - API: http://localhost:{config['api']['port']}")
        logger.info(f"  - Health: http://localhost:{config['api']['port']}/health")
        logger.info(f"  - Docs: http://localhost:{config['api']['port']}/docs")
        logger.info("  - Redis: localhost:6379")
        logger.info("  - Nginx: http://localhost:80")
    
    return success

def deploy_to_aws(config, logger):
    """Deploy to AWS ECS"""
    logger.info("Deploying to AWS...")
    
    # Build and push image to ECR
    ecr_repo = f"ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/{config['deployment']['docker']['image_name']}"
    
    commands = [
        ("aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com", "Login to ECR"),
        (f"docker tag {config['deployment']['docker']['image_name']}:latest {ecr_repo}:latest", "Tag image for ECR"),
        (f"docker push {ecr_repo}:latest", "Push image to ECR"),
        ("aws ecs register-task-definition --cli-input-json file://../aws/ecs-task-definition.json", "Register task definition"),
        ("aws ecs update-service --cluster decline-prediction-cluster --service decline-prediction --task-definition decline-prediction", "Update ECS service")
    ]
    
    for command, description in commands:
        if not run_command(command, description, logger, check=False):
            logger.error(f"AWS deployment failed at: {description}")
            return False
    
    logger.info("AWS deployment completed successfully")
    return True

def run_tests(logger):
    """Run test suite"""
    logger.info("Running tests...")
    
    test_commands = [
        ("python -m pytest ../tests/ -v", "Run unit tests"),
        ("python ../api/decline_api.py --test", "Test API startup")
    ]
    
    for command, description in test_commands:
        if not run_command(command, description, logger, check=False):
            logger.warning(f"Test failed: {description}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Deploy decline prediction API')
    parser.add_argument('--config', default='../config/decline_config.yaml', help='Configuration file')
    parser.add_argument('--target', choices=['local', 'aws'], default='local', help='Deployment target')
    parser.add_argument('--build-only', action='store_true', help='Only build Docker image')
    parser.add_argument('--test', action='store_true', help='Run tests after deployment')
    parser.add_argument('--skip-build', action='store_true', help='Skip building Docker image')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*50)
    logger.info("DECLINE PREDICTION API DEPLOYMENT")
    logger.info("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create deployment directories
    os.makedirs('../aws', exist_ok=True)
    os.makedirs('../logs', exist_ok=True)
    
    try:
        # Create deployment files
        create_requirements_file(logger)
        create_dockerfile(config, logger)
        create_docker_compose(config, logger)
        create_nginx_config(config, logger)
        create_aws_deployment_files(config, logger)
        
        if not args.skip_build:
            # Build Docker image
            if not build_docker_image(config, logger):
                logger.error("Docker build failed")
                return 1
        
        if args.build_only:
            logger.info("Build-only mode. Exiting.")
            return 0
        
        # Deploy based on target
        if args.target == 'local':
            if not run_local_deployment(config, logger):
                logger.error("Local deployment failed")
                return 1
        elif args.target == 'aws':
            if not deploy_to_aws(config, logger):
                logger.error("AWS deployment failed")
                return 1
        
        # Run tests if requested
        if args.test:
            run_tests(logger)
        
        logger.info("="*50)
        logger.info("DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        # Print useful information
        if args.target == 'local':
            logger.info("\nLocal Deployment URLs:")
            logger.info(f"API: http://localhost:{config['api']['port']}")
            logger.info(f"Health Check: http://localhost:{config['api']['port']}/health")
            logger.info(f"API Documentation: http://localhost:{config['api']['port']}/docs")
            logger.info("\nUseful Commands:")
            logger.info("docker-compose -f ../docker-compose.yml logs -f    # View logs")
            logger.info("docker-compose -f ../docker-compose.yml down       # Stop services")
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)