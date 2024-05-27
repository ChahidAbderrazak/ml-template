pipeline {
    agent any

    // triggers {
    //     pollSCM '* * * * *'
    // }

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh '''
                python3 -m venv ./venv
                ./venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Test') {
            steps {
                echo 'Testing..'
                sh '''
                cd tests
                ./venv/bin/python -m pytest
                '''
            }
        }
    }
}
