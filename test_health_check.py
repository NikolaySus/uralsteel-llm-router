#!/usr/bin/env python3
"""
Test script to verify gRPC health check functionality.
"""

import grpc
import sys
import os

SERVER_ADDRESS = os.environ.get('SERVER_ADDRESS', 'localhost:50051')
SECRET_KEY = os.environ.get('SECRET_KEY', '')
# Чтение доверенных сертификатов
with open("ca.crt", "rb") as f:
    TRUSTED_CERTS = f.read()
CREDS = grpc.ssl_channel_credentials(root_certificates=TRUSTED_CERTS)

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpc_health.v1 import health_pb2, health_pb2_grpc

def test_health_check():
    """Test the health check endpoint."""
    try:
        # Create a channel to the server
        # Use insecure channel for testing (adjust if your server uses secure connection)
        channel = grpc.secure_channel(SERVER_ADDRESS, CREDS)
        
        # Create a health check stub
        health_stub = health_pb2_grpc.HealthStub(channel)
        
        # Test general health check
        print("Testing general health check...")
        request = health_pb2.HealthCheckRequest()
        response = health_stub.Check(request)
        print(f"General health status: {health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)}")
        
        # Test specific service health check (LLM service)
        print("\nTesting LLM service health check...")
        request = health_pb2.HealthCheckRequest(service='llm.Llm')
        response = health_stub.Check(request)
        print(f"LLM service health status: {health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)}")
        
        # Test non-existent service (should return NOT_FOUND)
        print("\nTesting non-existent service health check...")
        request = health_pb2.HealthCheckRequest(service='nonexistent.service')
        try:
            response = health_stub.Check(request)
            print(f"Non-existent service status: {health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)}")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                print("Non-existent service correctly returned NOT_FOUND")
            else:
                print(f"Unexpected error: {e}")
        
        print("\nHealth check test completed successfully!")
        
    except grpc.RpcError as e:
        print(f"gRPC error: {e}")
        print(f"Error code: {e.code()}")
        print(f"Error details: {e.details()}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing gRPC Health Check...")
    success = test_health_check()
    sys.exit(0 if success else 1)