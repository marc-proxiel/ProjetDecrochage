# =============================================================================
#  install_docker_windows.ps1  —  Installer Docker Desktop sur Windows 10/11
#  Sprint 3 CISIA · Decrochage 4.0 · pour le J3 (modules 27 Docker + 28 compose)
# -----------------------------------------------------------------------------
#  À LANCER DANS UN POWERSHELL « EN TANT QU'ADMINISTRATEUR ».
#  Au besoin, autorise l'exécution du script pour CETTE session uniquement :
#     Unblock-File .\install_docker_windows.ps1        # enleve le blocage "fichier venu d'Internet"
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # NE PAS ecrire 'powershell' devant !
#  puis :  .\install_docker_windows.ps1
# =============================================================================

$ErrorActionPreference = "Stop"   # on s'arrête à la première vraie erreur

Write-Host "== Installation de Docker Desktop (Windows) ==" -ForegroundColor Cyan

# 0) Vérifier qu'on est bien administrateur (Docker + WSL en ont besoin).
$estAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $estAdmin) {
    Write-Error "Relance ce script dans un PowerShell ouvert EN ADMINISTRATEUR (clic droit > Exécuter en tant qu'administrateur)."
    exit 1
}

# 1) Déjà installé ? Si oui, on ne refait rien.
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Docker est déjà présent :" -ForegroundColor Green
    docker --version
    Write-Host "Si Docker Desktop est lancé (icône baleine fixe), tu es prêt pour le J3." -ForegroundColor Green
    exit 0
}

# 2) Activer WSL2 — c'est le moteur (« backend ») de Docker Desktop sous Windows.
#    `wsl --install --no-distribution` active les fonctionnalités nécessaires
#    (Sous-système Linux + Plateforme de machine virtuelle) sans installer Ubuntu.
Write-Host "-> Activation de WSL2 (backend de Docker)..." -ForegroundColor Yellow
try {
    wsl --install --no-distribution
} catch {
    # Repli pour de très anciennes versions de Windows 10 : on active les
    # fonctionnalités manuellement (un redémarrage sera nécessaire ensuite).
    Write-Host "   (méthode de repli : activation des fonctionnalités Windows)"
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Null
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Null
}

# 3) Installer Docker Desktop.
#    Méthode préférée : winget (gestionnaire de paquets intégré à Windows 11 /
#    Windows 10 récent). Sinon, on télécharge l'installeur officiel.
if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "-> Installation via winget..." -ForegroundColor Yellow
    winget install -e --id Docker.DockerDesktop `
        --accept-source-agreements --accept-package-agreements
} else {
    Write-Host "-> winget introuvable : téléchargement de l'installeur officiel..." -ForegroundColor Yellow
    $url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $exe = Join-Path $env:TEMP "DockerDesktopInstaller.exe"
    Invoke-WebRequest -Uri $url -OutFile $exe
    # Installation silencieuse + acceptation de la licence Docker Desktop.
    Start-Process -Wait -FilePath $exe -ArgumentList "install","--quiet","--accept-license"
}

# 4) Étapes finales (à faire à la main, elles ne se scriptent pas proprement).
Write-Host ""
Write-Host "== Dernières étapes ==" -ForegroundColor Cyan
Write-Host "1) REDÉMARRE le PC si Windows le demande (activation de WSL2)."
Write-Host "2) Ouvre 'Docker Desktop' (menu Démarrer) et attends que l'icône baleine soit FIXE."
Write-Host "3) Dans un NOUVEAU terminal, vérifie :" -ForegroundColor Green
Write-Host "      docker --version"
Write-Host "      docker compose version"
Write-Host "      docker run hello-world"
Write-Host ""
Write-Host "Si 'docker run hello-world' affiche un message de bienvenue, le J3 peut commencer." -ForegroundColor Green
Write-Host "Pré-requis matériel : la VIRTUALISATION doit être activée dans le BIOS/UEFI." -ForegroundColor DarkYellow
