pipeline {
    // agent any

    agent {
        docker { image 'python:3.8' }
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
                pip install -r requirements.txt

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
