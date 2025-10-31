#!/bin/bash

# Clean build
rm -rf ChaCha.app

# Create the app bundle structure first
mkdir -p ChaCha.app/Contents/MacOS

# Compile the Swift code
swiftc ChaCha.swift -o ChaCha.app/Contents/MacOS/ChaCha -framework SwiftUI -framework Metal -framework MetalKit -parse-as-library

# Create Info.plist
cat << EOF > ChaCha.app/Contents/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>ChaCha</string>
    <key>CFBundleIdentifier</key>
    <string>com.dh60.ChaCha</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
</dict>
</plist>
EOF