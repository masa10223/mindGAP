#!/bin/bash
# ELA解析一括実行シェルスクリプト
# 複数の解析を一括で実行し、詳細なログを取得します

set -e  # エラー時に停止

# 設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/batch_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_${TIMESTAMP}.log"
SUMMARY_FILE="${LOG_DIR}/batch_summary_${TIMESTAMP}.json"

# ログディレクトリの作成
mkdir -p "${LOG_DIR}"

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "${LOG_FILE}" >&2
}

# 実行結果の記録
declare -a JOB_RESULTS=()
declare -a JOB_NAMES=()
declare -a JOB_STATUS=()
declare -a JOB_DURATIONS=()

# 単一ジョブの実行
run_job() {
    local job_name="$1"
    local questionnaire_type="$2"
    local data_source="${3:-auto}"
    local hq25_threshold="${4:-3}"
    local notes="${5:-}"
    
    log "実行開始: ${job_name} (${questionnaire_type})"
    
    local start_time=$(date +%s)
    local cmd="python -m src.ela_pipeline --func ${questionnaire_type} --data_source ${data_source} --hq25_threshold ${hq25_threshold}"
    
    # 実行
    if eval "${cmd}" >> "${LOG_FILE}" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "✅ 完了: ${job_name} (${duration}秒)"
        JOB_RESULTS+=("success")
        JOB_STATUS+=("success")
        JOB_DURATIONS+=("${duration}")
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_error "❌ 失敗: ${job_name} (${duration}秒)"
        JOB_RESULTS+=("failed")
        JOB_STATUS+=("failed")
        JOB_DURATIONS+=("${duration}")
    fi
    
    JOB_NAMES+=("${job_name}")
}

# サマリーの生成
generate_summary() {
    local total_jobs=${#JOB_NAMES[@]}
    local successful_jobs=0
    local failed_jobs=0
    local total_duration=0
    
    # 成功・失敗の集計
    for status in "${JOB_STATUS[@]}"; do
        if [ "${status}" = "success" ]; then
            ((successful_jobs++))
        else
            ((failed_jobs++))
        fi
    done
    
    # 総実行時間の計算
    for duration in "${JOB_DURATIONS[@]}"; do
        total_duration=$((total_duration + duration))
    done
    
    # JSONサマリーの生成
    cat > "${SUMMARY_FILE}" << EOF
{
  "batch_id": "batch_${TIMESTAMP}",
  "start_time": "$(date -d "@${START_TIME}" '+%Y-%m-%dT%H:%M:%S')",
  "end_time": "$(date '+%Y-%m-%dT%H:%M:%S')",
  "total_duration_seconds": ${total_duration},
  "total_jobs": ${total_jobs},
  "successful_jobs": ${successful_jobs},
  "failed_jobs": ${failed_jobs},
  "success_rate": $(echo "scale=2; ${successful_jobs} * 100 / ${total_jobs}" | bc -l),
  "jobs": [
EOF
    
    # 各ジョブの詳細
    for i in "${!JOB_NAMES[@]}"; do
        local comma=""
        if [ $i -lt $((${#JOB_NAMES[@]} - 1)) ]; then
            comma=","
        fi
        
        cat >> "${SUMMARY_FILE}" << EOF
    {
      "name": "${JOB_NAMES[$i]}",
      "status": "${JOB_STATUS[$i]}",
      "duration_seconds": ${JOB_DURATIONS[$i]}
    }${comma}
EOF
    done
    
    cat >> "${SUMMARY_FILE}" << EOF
  ]
}
EOF
    
    # サマリーの表示
    log "=========================================="
    log "一括実行完了"
    log "総実行時間: ${total_duration}秒"
    log "成功: ${successful_jobs}件"
    log "失敗: ${failed_jobs}件"
    log "成功率: $(echo "scale=1; ${successful_jobs} * 100 / ${total_jobs}" | bc -l)%"
    log "ログファイル: ${LOG_FILE}"
    log "サマリーファイル: ${SUMMARY_FILE}"
    log "=========================================="
}

# メイン実行部分
main() {
    log "一括実行開始: $(date)"
    START_TIME=$(date +%s)
    
    # ここに実行したいジョブを追加
    # 形式: run_job "ジョブ名" "質問票タイプ" "データソース" "HQ-25閾値" "備考"
    
    # 例1: PHQ-9の解析
    run_job "PHQ9_all_analysis" "PHQ9_all" "csv" "3" "PHQ-9の全データ解析"
    
    # 例2: HQ-25の閾値比較
    run_job "HQ25_threshold2" "HQ-25+4" "excel" "2" "HQ-25閾値2での解析"
    run_job "HQ25_threshold3" "HQ-25+4" "excel" "3" "HQ-25閾値3での解析"
    
    # 例3: 他の質問票の解析
    # run_job "IPS22_analysis" "IPS-22" "excel" "4" "IPS-22の解析"
    # run_job "IAT_analysis" "IAT" "excel" "3" "IATの解析"
    # run_job "TACS22_analysis" "TACS-22" "excel" "4" "TACS-22の解析"
    
    # サマリーの生成
    generate_summary
}

# スクリプトの実行
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi