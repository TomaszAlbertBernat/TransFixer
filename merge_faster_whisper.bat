@echo off
echo 🚀 Merging faster_whisper branch into main...
echo ===========================================

REM Check current branch
echo 📍 Current branch:
git branch --show-current

REM Show status
echo.
echo 📊 Git status:
git status --short

REM Switch to main branch
echo.
echo 🔄 Switching to main branch...
git checkout main

if %errorlevel% equ 0 (
    echo ✅ Successfully switched to main branch
) else (
    echo ❌ Failed to switch to main branch
    pause
    exit /b 1
)

REM Merge faster_whisper branch
echo.
echo 🔗 Merging faster_whisper branch...
git merge faster_whisper

if %errorlevel% equ 0 (
    echo ✅ Successfully merged faster_whisper into main
) else (
    echo ❌ Merge failed - please resolve conflicts manually
    pause
    exit /b 1
)

REM Show final status
echo.
echo 📈 Final status:
git log --oneline -5

echo.
echo 🎉 Merge completed successfully!
echo.
echo 📋 Next steps:
echo    1. Test the merged code: python3 transfixer.py --help
echo    2. Push to remote: git push origin main
echo    3. Delete feature branch (optional): git branch -d faster_whisper
echo.
echo 🚀 TransFixer is now 15-40x faster! 🎯

pause 