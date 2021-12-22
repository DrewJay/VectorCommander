const fse = require('fs-extra');
const fs = require('fs');
const rimraf = require('rimraf');
const { exec } = require('child_process');

const modelPath = '../Progrezz/neural/model';
const analysisPath = '../Progrezz/neural/analysis';

// Remove folders so they can be reliably recreated.
if (fs.existsSync(modelPath)) { rimraf.sync(modelPath); }
if (fs.existsSync(analysisPath)) { rimraf.sync(analysisPath); }

// Recreate folders.
fs.mkdirSync(modelPath);
fs.mkdirSync(analysisPath);

// Copy files into destination and push to git.
fse.copy('run/analysis', analysisPath, (err) => {
    if (!err) {
        fse.copy('run/vae/1_output/model', modelPath, (_err) => {
            if (!_err) {
                console.log('File copying successful.');
                exec('cd ../Progrezz & aws s3 mv neural s3://arn:aws:s3:us-east-2:277059337208:accesspoint/virtual-cloud-s3-repository --recursive', (error, stdout, stderr) => {
                    console.log(error, stdout, stderr);
                });
            }
        });
    }
});

