pipeline {
    // agent any

    agent {
        dockerfile { filename 'Dockerfile' }
    }
    // triggers {
    //     pollSCM '* * * * *'
    // }

    stages {
       
        stage('Build') {
            steps {
                echo 'Building..'
                sh '''
                python --version
                '''
            }
        }

        stage('Test') {
            steps {
                echo 'Testing..'
                sh '''
                cd tests
                python -m pytest
                '''
            }
        }

    }
}
