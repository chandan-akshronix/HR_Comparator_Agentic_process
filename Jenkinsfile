pipeline {
    agent any
    
    environment {
        ACR_NAME = 'REPLACE_WITH_ACR_NAME' // Will be output from Terraform
        ACR_LOGIN_SERVER = '${ACR_NAME}.azurecr.io'
        IMAGE_NAME = 'hr-ai-agent'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        SONARQUBE_SERVER = 'SonarQube'
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
                    def scannerHome = tool 'sonar-scanner'
                    withSonarQubeEnv('sonar-scanner') {
                        sh """
                            ${scannerHome}/bin/sonar-scanner \
                              -Dsonar.projectKey=hr-ai-agent \
                              -Dsonar.projectName='HR AI Agent' \
                              -Dsonar.sources=. \
                              -Dsonar.python.coverage.reportPaths=coverage.xml \
                              -Dsonar.exclusions=**/*.pyc,**/__pycache__/**
                        """
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
                echo "✅ Quality Gate passed"
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
                    echo "Building Docker images..."
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
                    withCredentials([usernamePassword(credentialsId: 'acr-credentials', usernameVariable: 'ACR_USERNAME', passwordVariable: 'ACR_PASSWORD')]) {
                        sh """
                            echo \$ACR_PASSWORD | docker login ${ACR_LOGIN_SERVER} -u \$ACR_USERNAME --password-stdin
                            docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
                            docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:latest
                            docker logout ${ACR_LOGIN_SERVER}
                        """
                        echo "✅ Images pushed to Azure Container Registry"
                    }
                }
            }
        }
        
        /* COMMENTED OUT - Deploy to AKS stage
        stage('Deploy to AKS') {
            steps {
                script {
                    withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                        sh """
                            export KUBECONFIG=\$KUBECONFIG
                            
                            # Update deployment
                            kubectl set image deployment/hr-ai-agent \
                              hr-ai-agent=${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} \
                              -n hr-app
                            
                            # Wait for rollout
                            kubectl rollout status deployment/hr-ai-agent -n hr-app --timeout=5m
                            
                            # Verify deployment
                            kubectl get pods -n hr-app -l app=hr-ai-agent
                        """
                        echo "✅ Deployment successful"
                    }
                }
            }
        }
        */ // End of commented Deploy to AKS stage
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
            Status: Images pushed to ACR successfully
            Note: Deploy to AKS stage is commented out
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
