pipeline {
    agent any
    // triggers {
    //     pollSCM '* * * * *'
    // }

    stages {

        stage('Build ') {
            steps {
                echo 'Building docker container..'
                sh '''
                docker --version
                ls
                bash bash/build.sh
                '''
            }
        }
        stage('Test') {
            steps {
                echo 'Building..'
                sh '''
                bash bash/test.sh
                '''
            }
        }
        // stage('Build') {
        //     steps {
        //         echo 'Building..'
        //         sh '''
        //         python3 -m venv ./venv
        //         ./venv/bin/pip install -r requirements.txt
        //         '''
        //     }
        // }

        // stage('Test') {
        //     steps {
        //         echo 'Testing..'
        //         sh '''
        //         cd tests
        //         ./venv/bin/python -m pytest
        //         '''
        //     }
        // }

    }
}
