pipeline {
    agent any
    
    environment {
        ACR_NAME = 'hracrregistry'
        ACR_LOGIN_SERVER = 'hracrregistry.azurecr.io'
        IMAGE_NAME = 'hr-ai-agent'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        SONARQUBE_SERVER = 'http://4.213.4.181:9000'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "✅ Code checked out successfully"
            }
        }
        
        stage('SonarQube Analysis') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "📊 Starting SonarQube Code Quality Analysis..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    def scannerHome = tool 'sonar-scanner'
                    withSonarQubeEnv('sonar-scanner') {
                        sh """
                            ${scannerHome}/bin/sonar-scanner \
                              -Dsonar.projectKey=hr-ai-agent \
                              -Dsonar.projectName='HR AI Agent' \
                              -Dsonar.sources=. \
                              -Dsonar.python.coverage.reportPaths=coverage.xml \
                              -Dsonar.exclusions=**/*.pyc,**/migrations/**,**/__pycache__/**
                        """
                    }
                    echo "✅ SonarQube analysis submitted successfully"
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "⏳ Waiting for SonarQube Quality Gate result..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    timeout(time: 10, unit: 'MINUTES') {
                        def qg = waitForQualityGate()
                        echo "📊 Quality Gate Status: ${qg.status}"
                        if (qg.status != 'OK') {
                            error "❌ Quality Gate failed with status: ${qg.status}"
                        }
                    }
                    echo "✅ Quality Gate passed successfully"
                }
            }
        }
        
        stage('Trivy File System Scan') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "🔒 Running Trivy file system scan..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    sh """
                        trivy fs \
                          --severity HIGH,CRITICAL \
                          --exit-code 0 \
                          --format table \
                          .
                    """
                    echo "✅ Trivy file system scan completed"
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "🐳 Building Docker images for ACR..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    sh """
                        docker build -t ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} .
                        docker tag ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest
                    """
                    echo "✅ Docker images built successfully"
                }
            }
        }
        
        stage('Trivy Image Scan') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "🔒 Running Trivy Docker image security scan..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    sh """
                        trivy image \
                          --severity HIGH,CRITICAL \
                          --exit-code 0 \
                          --format table \
                          ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
                    """
                    echo "✅ Trivy image scan completed"
                }
            }
        }
        
        stage('Push to Azure Container Registry') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "☁️ Pushing Docker image to ACR..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    withCredentials([usernamePassword(credentialsId: 'acr-credentials', usernameVariable: 'ACR_USERNAME', passwordVariable: 'ACR_PASSWORD')]) {
                        sh """
                            echo \$ACR_PASSWORD | docker login ${ACR_LOGIN_SERVER} -u \$ACR_USERNAME --password-stdin
                            docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
                            docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest
                            docker logout ${ACR_LOGIN_SERVER}
                        """
                    }
                    echo "✅ Images pushed to Azure Container Registry"
                }
            }
        }
        
        stage('Deploy to AKS') {
            steps {
                script {
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "🚀 Deploying to AKS cluster..."
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                        sh """
                            export KUBECONFIG=\$KUBECONFIG
                            
                            # Update image tag in manifest
                            sed -i 's|image: ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:.*|image: ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}|g' ../k8s/05-ai-agent.yaml
                            
                            # Apply the manifest (creates or updates deployment)
                            kubectl apply -f ../k8s/05-ai-agent.yaml
                            
                            # Wait for rollout to complete
                            kubectl rollout status deployment/hr-ai-agent -n hr-app --timeout=5m
                            
                            # Verify deployment
                            echo "📋 Deployment Status:"
                            kubectl get pods -n hr-app -l app=hr-ai-agent
                        """
                    }
                    echo "✅ Deployment to AKS successful"
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
            sh "docker system prune -f || true"
        }
        success {
            echo """
            ========================================
            ✅ AI Agent Pipeline Successful!
            ========================================
            Image: ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
            Registry: Azure Container Registry
            Deployed to: hr-app namespace
            ========================================
            """
        }
        failure {
            echo """
            ========================================
            ❌ AI Agent Pipeline Failed!
            ========================================
            Check the logs above for details
            ========================================
            """
        }
    }
}
