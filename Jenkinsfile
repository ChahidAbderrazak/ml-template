pipeline {
    // agent any

    agent {
        // docker { image 'python:3.8' }
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
