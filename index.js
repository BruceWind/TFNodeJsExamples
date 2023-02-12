// // Adds the CPU backend.
// import '@tensorflow/tfjs-backend-cpu';
// // Import @tensorflow/tfjs-core
// import * as tf from '@tensorflow/tfjs-core';
// // Import @tensorflow/tfjs-tflite.
// import * as tflite from '@tensorflow/tfjs-tflite';

// Adds the CPU backend.
import '@tensorflow/tfjs-backend-cpu';
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs-node';
// const tf = require('@tensorflow/tfjs-node');
// Import @tensorflow/tfjs-tflite.
import * as tflite from 'tfjs-tflite-node';


import fs from 'fs';

setTimeout(async () => {

  // Load a TFLite model, which is created by yahoo.
  const YAHOO_NSFW_MODEL_URL = './nsfw.tflite';
  const tfliteModel = await tflite.loadTFLiteModel(YAHOO_NSFW_MODEL_URL, { numThreads: 1 });


  // a non-copyright porn image is downlodd from pornpen.ai
  const localPronImg = './images/porn_ai-generated.jpeg';
  const localNormalFaceImg = './images/avatar_ai_generated.jpeg';

  // Prepare input tensors.
  let img = tf.node.decodeJpeg(new Uint8Array(fs.readFileSync(localPronImg)));
  let input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);


  //resize image to suit tensorflow.
  const alignCorners = false;
  // Resize the cropped images to be [1,224,224,3]
  let imageResize = tf.image.resizeBilinear(
    input,
    [224, 224],
    alignCorners
  );


  // // Run inference and get output tensors.
  let outputTensor = tfliteModel.predict(imageResize);

  let predictedScore = outputTensor.arraySync()[0][0];
  console.log('pron image recognized result: '+predictedScore);


  // start to regonize a normal face image.

  img = tf.node.decodeJpeg(new Uint8Array(fs.readFileSync(localNormalFaceImg)));
  input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
  imageResize = tf.image.resizeBilinear(
    input,
    [224, 224],
    alignCorners
  );

  outputTensor = tfliteModel.predict(imageResize);

  predictedScore = outputTensor.arraySync()[0][0];
  console.log('a normal image recognized result: '+predictedScore);


  //to relase memory, or it may cause memory leak.
  tf.dispose();


}, 100);

