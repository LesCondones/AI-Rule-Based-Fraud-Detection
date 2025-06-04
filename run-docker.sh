#!/bin/bash

# AI Fraud Detection - Docker Setup Script
echo "🐳 AI Fraud Detection Docker Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is installed and running"

# Build and run the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting the application..."
docker-compose up -d

echo ""
echo "🎉 Success! Your AI Fraud Detection system is running in Docker!"
echo ""
echo "📍 Access the application at: http://localhost:5000"
echo ""
echo "📊 Useful Docker commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop app:      docker-compose down"
echo "   Restart:       docker-compose restart"
echo "   Shell access:  docker-compose exec fraud-detection bash"
echo ""
echo "🔍 Checking container status..."
docker-compose ps

# Wait a moment and check if the service is healthy
sleep 10
echo ""
echo "🏥 Health check..."
if curl -f http://localhost:5000/ &> /dev/null; then
    echo "✅ Application is healthy and responding!"
    echo "🌐 Open your browser to: http://localhost:5000"
else
    echo "⚠️  Application might still be starting up..."
    echo "   Check logs with: docker-compose logs -f"
fi