#!/bin/bash

# Check if running in the desired directory
echo "Generating project structure in: $(pwd)"
echo "Continue? (y/n)"
read -r response

if [[ "$response" != "y" ]]; then
    echo "Operation cancelled"
    exit 1
fi

# Create main project directories
directories=(
    # Frontend structure
    "frontend/public/assets/images"
    "frontend/public/assets/styles"
    "frontend/public/assets/fonts"
    "frontend/src/components/Header"
    "frontend/src/components/Footer"
    "frontend/src/components/Article"
    "frontend/src/pages"
    "frontend/src/layouts"
    "frontend/src/utils"
    
    # Backend structure
    "backend/src/controllers"
    "backend/src/middleware"
    "backend/src/routes"
    "backend/src/services"
    "backend/src/utils"
    "backend/config"
    
    # DITA structure
    "dita/maps"
    "dita/topics/articles"
    "dita/topics/journals"
    "dita/topics/abstracts"
    "dita/metadata"
    "dita/templates"
    
    # Middleware structure
    "middleware/dita-processor/transformers"
    "middleware/dita-processor/assemblers"
    "middleware/xslt/pdf"
    "middleware/xslt/html"
    "middleware/config"
    
    # Database structure
    "database/migrations"
    "database/seeds"
    "database/schemas"
    
    # Distribution structure
    "dist/html"
    "dist/pdf"
    "dist/assets"
    
    # Test structure
    "tests/frontend"
    "tests/backend"
    "tests/integration"
    
    # Documentation structure
    "docs/api"
    "docs/setup"
    "docs/workflows"
)

# Create directories
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "Created directory: $dir"
done

# Create root level files
touch README.md
touch .gitignore
touch docker-compose.yml

# Create package.json files
touch frontend/package.json
touch frontend/README.md
touch backend/package.json
touch backend/README.md

# Create database configuration
touch backend/config/database.js

# Create initial DITA map file
touch dita/maps/root-map.ditamap

echo "Project structure has been generated successfully!"

# Print summary
echo -e "\nCreated files:"
find . -type f -exec echo "File: {}" \;

echo -e "\nCreated directories:"
find . -type d -not -name "." -exec echo "Directory: {}" \;

echo -e "\nDone! Your project structure is ready."
