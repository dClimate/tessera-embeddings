{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/tmp/ray/session_latest/logs/raylet.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/raylet",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/gcs_server.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/gcs_server",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/dashboard.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/dashboard",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/monitor.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/monitor",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/worker-*.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/workers",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/worker-*.err",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/actors",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/ram_poll.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/ram_poll",
            "timezone": "UTC"
          },
          {
            "file_path": "/tmp/ray/session_latest/logs/**/*.log",
            "log_group_name": "__LOG_GROUP__",
            "log_stream_name": "__INSTANCE_ID__/other",
            "blacklist": "ram_poll.log",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
