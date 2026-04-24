#!/bin/bash
# Sephora Recommender - Google Cloud Run Quick Deploy
# Bu script'i çalıştırmadan önce GCP_DEPLOY_GUIDE.md'yi oku

set -e  # Stop on error

echo "🚀 Sephora Recommender - Google Cloud Deploy"
echo "=============================================="

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================
# 1. PROJE AYARLARI
# ============================================

echo -e "\n${BLUE}[1/6] Proje ayarları...${NC}"

# Proje ID'sini değiştir
PROJECT_ID="sephora-recommender"
REGION="europe-west1"

echo "Proje ID: $PROJECT_ID"
echo "Region: $REGION"

# Proje seç
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# ============================================
# 2. API'LERI AKTİFLEŞTİR
# ============================================

echo -e "\n${BLUE}[2/6] API'ler aktifleştiriliyor...${NC}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com

echo -e "${GREEN}✓ API'ler aktif${NC}"

# ============================================
# 3. DOSYA KONTROLÜ
# ============================================

echo -e "\n${BLUE}[3/6] Dosyalar kontrol ediliyor...${NC}"

# Gerekli dosyaları kontrol et
files_ok=true

if [ ! -f "Dockerfile.api" ]; then
    echo -e "${RED}✗ Dockerfile.api bulunamadı${NC}"
    files_ok=false
fi

if [ ! -f "Dockerfile.ui" ]; then
    echo -e "${RED}✗ Dockerfile.ui bulunamadı${NC}"
    files_ok=false
fi

if [ ! -f "outputs/models/config.json" ]; then
    echo -e "${RED}✗ outputs/models/config.json bulunamadı${NC}"
    files_ok=false
fi

if [ ! -f "outputs/models/product_concern_embeddings.pkl" ]; then
    echo -e "${RED}✗ outputs/models/product_concern_embeddings.pkl bulunamadı${NC}"
    files_ok=false
fi

if [ ! -f "data/processed/ml_scoring_table.parquet" ]; then
    echo -e "${RED}✗ data/processed/ml_scoring_table.parquet bulunamadı${NC}"
    files_ok=false
fi

if [ "$files_ok" = false ]; then
    echo -e "\n${RED}Eksik dosyalar var. Önce train.py çalıştır.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Tüm dosyalar mevcut${NC}"

# ============================================
# 4. API DEPLOY
# ============================================

echo -e "\n${BLUE}[4/6] API deploy ediliyor...${NC}"
echo "Bu işlem 5-10 dakika sürebilir ☕"

gcloud run deploy sephora-api \
  --source . \
  --dockerfile Dockerfile.api \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --quiet

echo -e "${GREEN}✓ API deploy edildi${NC}"

# API URL'ini al
API_URL=$(gcloud run services describe sephora-api \
  --region $REGION \
  --format 'value(status.url)')

echo "API URL: $API_URL"

# ============================================
# 5. API TEST
# ============================================

echo -e "\n${BLUE}[5/6] API test ediliyor...${NC}"

# Health check
if curl -s "$API_URL/health" | grep -q "ok"; then
    echo -e "${GREEN}✓ Health check başarılı${NC}"
else
    echo -e "${RED}✗ Health check başarısız${NC}"
    exit 1
fi

# Concerns
CONCERNS=$(curl -s "$API_URL/concerns" | jq -r '.concerns | length')
echo -e "${GREEN}✓ Concerns yüklendi: $CONCERNS adet${NC}"

# ============================================
# 6. UI DEPLOY
# ============================================

echo -e "\n${BLUE}[6/6] UI deploy ediliyor...${NC}"
echo "Bu işlem 5-10 dakika sürebilir ☕"

gcloud run deploy sephora-ui \
  --source . \
  --dockerfile Dockerfile.ui \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 5 \
  --min-instances 0 \
  --set-env-vars API_URL=$API_URL \
  --quiet

echo -e "${GREEN}✓ UI deploy edildi${NC}"

# UI URL'ini al
UI_URL=$(gcloud run services describe sephora-ui \
  --region $REGION \
  --format 'value(status.url)')

# ============================================
# SONUÇ
# ============================================

echo ""
echo "=============================================="
echo -e "${GREEN}🎉 DEPLOY TAMAMLANDI!${NC}"
echo "=============================================="
echo ""
echo "📡 API URL:  $API_URL"
echo "🎨 UI URL:   $UI_URL"
echo ""
echo "Test et:"
echo "  curl $API_URL/health"
echo "  open $UI_URL"
echo ""
echo "Loglar:"
echo "  gcloud run logs tail sephora-api --region $REGION"
echo "  gcloud run logs tail sephora-ui --region $REGION"
echo ""
echo "=============================================="
