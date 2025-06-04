# 🐳 Docker Setup Guide for AI Fraud Detection

## Step 1: Install Docker Desktop

1. **Download Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. **Install** following the installer instructions
3. **Start Docker Desktop** - you'll see a whale icon in your system tray when it's running

## Step 2: Verify Docker Installation

Open Terminal and run:
```bash
docker --version
docker-compose --version
```

You should see version numbers if Docker is properly installed.

## Step 3: Build and Run Your Fraud Detection System

```bash
# Navigate to your project directory
cd "/Users/lesartis/Desktop/Code Projects/AI Rule Based Fraud Detection"

# Option A: Use the automated script
./run-docker.sh

# Option B: Manual commands
docker-compose build    # Build the container (first time only)
docker-compose up       # Start the application
```

## Step 4: Access Your Application

- **Open your browser** to: http://localhost:5000
- **Upload bank statements** and analyze for fraud
- **All ML features** work without any Python installation issues!

## 🔧 Useful Docker Commands

```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# Access container shell (for debugging)
docker-compose exec fraud-detection bash

# View running containers
docker-compose ps

# Rebuild after code changes
docker-compose build --no-cache
```

## 🎯 What This Solves

| Issue | Solution |
|-------|----------|
| ❌ Bus errors | ✅ Isolated Linux environment |
| ❌ Library conflicts | ✅ Pre-tested dependencies |
| ❌ Python version issues | ✅ Consistent Python 3.11 |
| ❌ ML package conflicts | ✅ Stable ML library versions |
| ❌ Complex setup | ✅ One command to run |

## 📁 How It Works

1. **Docker creates a Linux container** on your Mac
2. **All Python libraries** are installed in the container
3. **Your app runs** in this isolated environment
4. **Port 5000 is mapped** so you can access it at localhost:5000
5. **File uploads** are saved to mounted volumes

## 🚨 Troubleshooting

### Container won't start:
```bash
docker-compose logs -f
```

### Port already in use:
```bash
# Kill any process using port 5000
sudo lsof -ti:5000 | xargs kill -9
```

### Want to start fresh:
```bash
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

### Check container health:
```bash
curl http://localhost:5000/health
```

## 🎉 Success Indicators

✅ Docker Desktop is running (whale icon visible)  
✅ `docker-compose up` runs without errors  
✅ Browser shows the app at http://localhost:5000  
✅ You can upload files and see analysis results  
✅ No more bus errors or library conflicts!  

## 🔄 Making Changes

If you modify the Python code:
1. Stop the container: `docker-compose down`
2. Rebuild: `docker-compose build`
3. Start again: `docker-compose up`

## 💡 Pro Tips

- Keep Docker Desktop running when using the app
- The container automatically restarts if it crashes
- All your uploads are saved to the `uploads/` folder
- Results are saved to the `results/` folder
- You can run multiple containers on different ports if needed

Happy fraud detecting! 🕵️‍♂️