package org.opencv.samples.facedetect;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import android.content.Intent;
import android.provider.MediaStore;
import androidx.annotation.Nullable;
import android.net.Uri;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BOWImgDescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class FaceDetectActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "LogoDetect";

    private CameraBridgeViewBase cameraView;
    private ImageView descriptorView;
    private Mat lastFrame;

    // SIFT + BoW
    private SIFT sift;
    private DescriptorMatcher matcher;
    private BOWImgDescriptorExtractor bowExtractor;
    private Mat vocabMat;

    private static final int PICK_IMAGE_REQUEST = 1;

    // Custom SVM weights (including bias row)
    private float[][] weights;
    private int nClass;

    private final String[] labels = new String[]{
            "Adidas","Apple","BMW","Citroen","Cocacola","DHL",
            "Fedex","Ferrari","Ford","Google","HP","Heineken",
            "Intel","McDonalds","Mini","Nbc","Nike","Pepsi",
            "Porsche","Puma","RedBull","Sprite","Starbucks",
            "Texaco","Unicef","Vodafone","Yahoo"
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV init failed", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        sift = SIFT.create();
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_SL2);
        bowExtractor = new BOWImgDescriptorExtractor(sift, matcher);
        vocabMat = loadCsvToMat(R.raw.vocab);
        bowExtractor.setVocabulary(vocabMat);

        try {
            weights = load2dCsv(R.raw.weight_matrix);
            nClass = weights[0].length;
        } catch (IOException e) {
            Log.e(TAG, "Failed to load SVM weights", e);
            Toast.makeText(this, "Failed to load weights", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        setContentView(R.layout.face_detect_surface_view);
        cameraView = findViewById(R.id.fd_activity_surface_view);
        cameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        descriptorView = findViewById(R.id.descriptorView);

        ((Button)findViewById(R.id.btnCapture)).setOnClickListener(v -> {
            if (lastFrame != null) runSiftBowPredict(lastFrame);
        });
        findViewById(R.id.btnUpload).setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(
                    Intent.createChooser(intent, "Select Logo Image"),
                    PICK_IMAGE_REQUEST);
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            try {
                Bitmap bmp = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                Mat mat = new Mat();
                Utils.bitmapToMat(bmp.copy(Bitmap.Config.ARGB_8888, true), mat);
                runSiftBowPredict(mat);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Mat loadCsvToMat(int rawRes) {
        try (InputStream is = getResources().openRawResource(rawRes);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            List<float[]> rows = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                String[] tok = line.split(",");
                float[] r = new float[tok.length];
                for (int i = 0; i < tok.length; i++) r[i] = Float.parseFloat(tok[i]);
                rows.add(r);
            }
            int R = rows.size(), C = rows.get(0).length;
            Mat m = new Mat(R, C, CvType.CV_32F);
            float[] flat = new float[R * C];
            for (int i = 0; i < R; i++) System.arraycopy(rows.get(i), 0, flat, i * C, C);
            m.put(0, 0, flat);
            return m;
        } catch (Exception ex) {
            throw new RuntimeException("loadCsvToMat failed", ex);
        }
    }

    private float[][] load2dCsv(int rawRes) throws IOException {
        InputStream is = getResources().openRawResource(rawRes);
        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        List<float[]> rows = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            String[] tok = line.split(",");
            float[] r = new float[tok.length];
            for (int i = 0; i < tok.length; i++) r[i] = Float.parseFloat(tok[i]);
            rows.add(r);
        }
        br.close();
        return rows.toArray(new float[rows.size()][]);
    }

    @Override public void onResume() { super.onResume(); cameraView.enableView(); }
    @Override public void onPause()  { super.onPause();  cameraView.disableView(); }
    @Override public void onDestroy(){ super.onDestroy(); cameraView.disableView(); }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraView);
    }
    @Override public void onCameraViewStarted(int w,int h){}
    @Override public void onCameraViewStopped(){}
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (lastFrame!=null) lastFrame.release();
        lastFrame = inputFrame.rgba().clone();
        return inputFrame.rgba();
    }

    private void runSiftBowPredict(Mat frame) {
        // preprocess
        Mat gray = new Mat(), eq = new Mat(), patch = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(gray, eq); gray.release();
        Imgproc.resize(eq, patch, new Size(128,128)); eq.release();

        // SIFT
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat descs = new Mat();
        sift.detectAndCompute(patch, new Mat(), kps, descs);
        if (descs.empty()) {
            patch.release(); kps.release(); descs.release();
            runOnUiThread(() -> Toast.makeText(this, "No features", Toast.LENGTH_SHORT).show());
            return;
        }

        // draw keypoints
        Mat kpImg = new Mat();
        Features2d.drawKeypoints(patch, kps, kpImg,
                new Scalar(0,255,0), Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
        Bitmap bmp = Bitmap.createBitmap(kpImg.cols(), kpImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(kpImg, bmp);
        runOnUiThread(() -> descriptorView.setImageBitmap(bmp));
        kpImg.release();

        // compute raw BoW
//        Mat bowDesc = computeRawBow(descs);
        Mat bowDesc = new Mat();
        bowExtractor.compute(patch, kps, bowDesc);
        float[] fv_bow = new float[bowDesc.cols()];
        Core.normalize(bowDesc, bowDesc, 1.0, 0.0, Core.NORM_L2);
        bowDesc.get(0, 0, fv_bow);
        Log.d(TAG, "BoW: " + Arrays.toString(fv_bow));

        // build feature vector + bias
        int D = bowDesc.cols();
        float[] fv = new float[D+1];
        bowDesc.get(0, 0, fv);
        fv[D] = 1f;  // bias term

        // normalize the augmented vector
        float norm = 0f;
        for (float v : fv) norm += v*v;
        norm = (float)Math.sqrt(norm);
        for (int i = 0; i < fv.length; i++) fv[i] /= norm;


        // compute scores for each class
        float[] scores = new float[nClass];
        for (int c = 0; c < nClass; c++) {
            float s = 0f;
            for (int j = 0; j < fv.length; j++) {
                s += fv[j] * weights[j][c];
            }
            scores[c] = s;
        }

        // find top 3 predictions
        int[] topIndices = new int[3];
        float[] topScores = new float[3];
        for (int i = 0; i < 3; i++) {
            float maxScore = Float.NEGATIVE_INFINITY;
            int maxIdx = -1;
            for (int c = 0; c < nClass; c++) {
                boolean alreadyPicked = false;
                for (int k = 0; k < i; k++) {
                    if (topIndices[k] == c) {
                        alreadyPicked = true;
                        break;
                    }
                }
                if (!alreadyPicked && scores[c] > maxScore) {
                    maxScore = scores[c];
                    maxIdx = c;
                }
            }
            topIndices[i] = maxIdx;
            topScores[i] = maxScore;
        }

        // log all class scores
        StringBuilder sb = new StringBuilder("Scores: ");
        for (int i = 0; i < nClass; i++) {
            sb.append(labels[i])
                    .append("=")
                    .append(String.format(Locale.US, "%.4f", scores[i]))
                    .append(i + 1 < nClass ? ", " : "");
        }
        Log.d(TAG, sb.toString());

        // build top 3 result string
        StringBuilder topSb = new StringBuilder("Top 3 Predictions:\n");
        for (int i = 0; i < 3; i++) {
            topSb.append(String.format(Locale.US, "%d. %s (%.2f)\n",
                    i + 1, labels[topIndices[i]], topScores[i]));
        }

        // show top 3 on UI
        runOnUiThread(() -> {
            new AlertDialog.Builder(this)
                    .setTitle("Top 3 Predictions")
                    .setMessage(topSb.toString())
                    .setPositiveButton("OK", null)
                    .show();
        });

        patch.release(); kps.release(); descs.release(); bowDesc.release();
    }


    private Mat computeRawBow(Mat descs) {
        matcher.clear();
        matcher.add(Collections.singletonList(vocabMat));
        int K=vocabMat.rows(); float[] hist=new float[K];
        List<MatOfDMatch> matches=new ArrayList<>();
        matcher.knnMatch(descs, matches, 1);
        for (MatOfDMatch mdm: matches) {
            DMatch[] dm = mdm.toArray();
            if (dm.length>0) hist[dm[0].trainIdx]++; }
        Mat raw=new Mat(1,K,CvType.CV_32F);
        raw.put(0,0,hist);
        return raw;
    }
}


