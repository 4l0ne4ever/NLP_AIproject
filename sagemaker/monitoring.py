"""
CloudWatch Monitoring and Logging for SageMaker Resources

This module provides comprehensive monitoring and logging capabilities for:
- Training job metrics and logs
- Endpoint performance monitoring
- Cost tracking and alerts
- Custom dashboards and alarms
"""

import boto3
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

from config import SageMakerConfigManager

logger = logging.getLogger(__name__)


class SageMakerMonitoring:
    """Comprehensive monitoring for SageMaker resources"""
    
    def __init__(self, config: SageMakerConfigManager = None):
        self.config = config or SageMakerConfigManager()
        
        # AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.config.s3_config.region)
        self.logs_client = boto3.client('logs', region_name=self.config.s3_config.region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config.s3_config.region)
        
        # Metric namespaces
        self.sagemaker_namespace = '/aws/sagemaker'
        self.custom_namespace = 'StrangerThings/SageMaker'
    
    def create_dashboard(self, dashboard_name: str = "StrangerThings-SageMaker") -> str:
        """Create a CloudWatch dashboard for monitoring SageMaker resources"""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", {"stat": "Sum"}],
                            ["AWS/SageMaker", "ModelLatency", "EndpointName", {"stat": "Average"}],
                            ["AWS/SageMaker", "InvocationsPerInstance", "EndpointName", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.config.s3_config.region,
                        "title": "Endpoint Performance",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "EndpointName", {"stat": "Average"}],
                            ["AWS/SageMaker", "MemoryUtilization", "EndpointName", {"stat": "Average"}],
                            ["AWS/SageMaker", "GPUUtilization", "EndpointName", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.config.s3_config.region,
                        "title": "Resource Utilization",
                        "period": 300
                    }
                },
                {
                    "type": "log",
                    "x": 0,
                    "y": 6,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "query": f"SOURCE '/aws/sagemaker/Endpoints'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                        "region": self.config.s3_config.region,
                        "title": "Recent Endpoint Errors",
                        "view": "table"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "TrainingTime", "TrainingJobName", {"stat": "Average"}],
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.config.s3_config.region,
                        "title": "Training Job Metrics",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 12,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [self.custom_namespace, "InferenceLatency", "Model", {"stat": "Average"}],
                            [self.custom_namespace, "InferenceCount", "Model", {"stat": "Sum"}],
                            [self.custom_namespace, "ErrorRate", "Model", {"stat": "Average"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.config.s3_config.region,
                        "title": "Custom Application Metrics",
                        "period": 300
                    }
                }
            ]
        }
        
        try:
            response = self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            dashboard_url = f"https://{self.config.s3_config.region}.console.aws.amazon.com/cloudwatch/home?region={self.config.s3_config.region}#dashboards:name={dashboard_name}"
            
            logger.info(f"✅ Created CloudWatch dashboard: {dashboard_name}")
            logger.info(f"🔗 Dashboard URL: {dashboard_url}")
            
            return dashboard_url
            
        except Exception as e:
            logger.error(f"❌ Failed to create dashboard: {e}")
            raise
    
    def create_alarms(self, endpoint_name: str) -> List[str]:
        """Create CloudWatch alarms for endpoint monitoring"""
        
        alarms_created = []
        
        # High error rate alarm
        error_alarm_name = f"SageMaker-HighErrorRate-{endpoint_name}"
        try:
            self.cloudwatch.put_metric_alarm(
                AlarmName=error_alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=2,
                MetricName='4XXError',
                Namespace='AWS/SageMaker',
                Period=300,
                Statistic='Sum',
                Threshold=10.0,
                ActionsEnabled=True,
                AlarmActions=[
                    # Add SNS topic ARN here if you want notifications
                ],
                AlarmDescription=f'High error rate for SageMaker endpoint {endpoint_name}',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    },
                ],
                Unit='Count',
                TreatMissingData='notBreaching'
            )
            alarms_created.append(error_alarm_name)
            logger.info(f"✅ Created error rate alarm: {error_alarm_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create error alarm: {e}")
        
        # High latency alarm
        latency_alarm_name = f"SageMaker-HighLatency-{endpoint_name}"
        try:
            self.cloudwatch.put_metric_alarm(
                AlarmName=latency_alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=3,
                MetricName='ModelLatency',
                Namespace='AWS/SageMaker',
                Period=300,
                Statistic='Average',
                Threshold=5000.0,  # 5 seconds
                ActionsEnabled=True,
                AlarmDescription=f'High latency for SageMaker endpoint {endpoint_name}',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    },
                ],
                Unit='Milliseconds',
                TreatMissingData='notBreaching'
            )
            alarms_created.append(latency_alarm_name)
            logger.info(f"✅ Created latency alarm: {latency_alarm_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create latency alarm: {e}")
        
        # Low invocation count alarm (could indicate issues)
        invocation_alarm_name = f"SageMaker-LowInvocations-{endpoint_name}"
        try:
            self.cloudwatch.put_metric_alarm(
                AlarmName=invocation_alarm_name,
                ComparisonOperator='LessThanThreshold',
                EvaluationPeriods=4,
                MetricName='Invocations',
                Namespace='AWS/SageMaker',
                Period=900,  # 15 minutes
                Statistic='Sum',
                Threshold=1.0,
                ActionsEnabled=True,
                AlarmDescription=f'Very low invocation count for endpoint {endpoint_name}',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    },
                ],
                Unit='Count',
                TreatMissingData='breaching'  # Treat missing data as breach
            )
            alarms_created.append(invocation_alarm_name)
            logger.info(f"✅ Created invocation alarm: {invocation_alarm_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create invocation alarm: {e}")
        
        return alarms_created
    
    def get_endpoint_metrics(self, endpoint_name: str, 
                           start_time: datetime = None,
                           end_time: datetime = None) -> Dict:
        """Get comprehensive metrics for an endpoint"""
        
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()
        
        metrics = {}
        
        # Define metrics to retrieve
        metric_queries = [
            ('Invocations', 'Sum'),
            ('ModelLatency', 'Average'),
            ('4XXError', 'Sum'),
            ('5XXError', 'Sum'),
            ('InvocationsPerInstance', 'Average'),
            ('CPUUtilization', 'Average'),
            ('MemoryUtilization', 'Average'),
            ('GPUUtilization', 'Average')
        ]
        
        for metric_name, statistic in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/SageMaker',
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,  # 1 hour intervals
                    Statistics=[statistic]
                )
                
                datapoints = response['Datapoints']
                if datapoints:
                    # Sort by timestamp and get latest values
                    datapoints.sort(key=lambda x: x['Timestamp'])
                    
                    metrics[metric_name] = {
                        'latest_value': datapoints[-1][statistic],
                        'average': sum(dp[statistic] for dp in datapoints) / len(datapoints),
                        'datapoints': len(datapoints),
                        'unit': datapoints[-1].get('Unit', 'None')
                    }
                else:
                    metrics[metric_name] = {
                        'latest_value': 0,
                        'average': 0,
                        'datapoints': 0,
                        'unit': 'None'
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {metric_name}: {e}")
                metrics[metric_name] = {'error': str(e)}
        
        return metrics
    
    def get_training_job_logs(self, job_name: str, 
                            lines: int = 100) -> List[str]:
        """Get recent logs from a training job"""
        
        log_group_name = f'/aws/sagemaker/TrainingJobs'
        log_stream_name = f'{job_name}/algo-1-{int(time.time())}'
        
        try:
            # First, list log streams for this training job
            streams_response = self.logs_client.describe_log_streams(
                logGroupName=log_group_name,
                logStreamNamePrefix=job_name,
                orderBy='LastEventTime',
                descending=True,
                limit=5
            )
            
            log_lines = []
            
            for stream in streams_response['logStreams']:
                try:
                    events_response = self.logs_client.get_log_events(
                        logGroupName=log_group_name,
                        logStreamName=stream['logStreamName'],
                        limit=lines,
                        startFromHead=False
                    )
                    
                    for event in events_response['events']:
                        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                        log_lines.append(f"[{timestamp}] {event['message']}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get logs from stream {stream['logStreamName']}: {e}")
                    continue
            
            return log_lines[-lines:] if log_lines else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get training logs: {e}")
            return [f"Error retrieving logs: {str(e)}"]
    
    def get_endpoint_logs(self, endpoint_name: str, 
                         lines: int = 100,
                         hours_back: int = 24) -> List[str]:
        """Get recent logs from an endpoint"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Convert to milliseconds since epoch
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        
        try:
            # Query CloudWatch Logs Insights
            query = f"""
            fields @timestamp, @message
            | filter @message like /{endpoint_name}/
            | sort @timestamp desc
            | limit {lines}
            """
            
            query_response = self.logs_client.start_query(
                logGroupName='/aws/sagemaker/Endpoints',
                startTime=start_time_ms,
                endTime=end_time_ms,
                queryString=query
            )
            
            query_id = query_response['queryId']
            
            # Wait for query to complete
            max_wait = 30  # seconds
            wait_time = 0
            
            while wait_time < max_wait:
                result_response = self.logs_client.get_query_results(queryId=query_id)
                
                if result_response['status'] == 'Complete':
                    log_lines = []
                    for result in result_response['results']:
                        timestamp_field = next((r for r in result if r['field'] == '@timestamp'), None)
                        message_field = next((r for r in result if r['field'] == '@message'), None)
                        
                        if timestamp_field and message_field:
                            log_lines.append(f"[{timestamp_field['value']}] {message_field['value']}")
                    
                    return log_lines
                
                elif result_response['status'] == 'Failed':
                    return [f"Query failed: {result_response.get('statistics', {}).get('recordsMatched', 'Unknown error')}"]
                
                time.sleep(2)
                wait_time += 2
            
            return [f"Query timeout after {max_wait} seconds"]
            
        except Exception as e:
            logger.error(f"❌ Failed to get endpoint logs: {e}")
            return [f"Error retrieving logs: {str(e)}"]
    
    def publish_custom_metric(self, metric_name: str, value: float, 
                            dimensions: Dict[str, str] = None,
                            unit: str = 'None') -> bool:
        """Publish custom metrics to CloudWatch"""
        
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.custom_namespace,
                MetricData=[metric_data]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish metric {metric_name}: {e}")
            return False
    
    def get_cost_metrics(self, service: str = 'AmazonSageMaker',
                        days_back: int = 30) -> Dict:
        """Get cost metrics for SageMaker usage"""
        
        try:
            # Use Cost Explorer API
            cost_client = boto3.client('ce', region_name='us-east-1')  # Cost Explorer is only in us-east-1
            
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days_back)
            
            response = cost_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': [service]
                    }
                }
            )
            
            cost_data = {}
            total_cost = 0
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                
                for group in result['Groups']:
                    service_name = group['Keys'][0]
                    amount = float(group['Metrics']['BlendedCost']['Amount'])
                    
                    if service_name not in cost_data:
                        cost_data[service_name] = {'daily_costs': [], 'total': 0}
                    
                    cost_data[service_name]['daily_costs'].append({
                        'date': date,
                        'cost': amount
                    })
                    cost_data[service_name]['total'] += amount
                    total_cost += amount
            
            return {
                'total_cost': total_cost,
                'period_days': days_back,
                'services': cost_data,
                'daily_average': total_cost / days_back if days_back > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get cost metrics: {e}")
            return {'error': str(e)}
    
    def generate_monitoring_report(self, endpoint_name: str = None,
                                 training_job_name: str = None) -> str:
        """Generate a comprehensive monitoring report"""
        
        report_lines = []
        report_lines.append("🔍 SageMaker Monitoring Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Endpoint metrics
        if endpoint_name:
            report_lines.append(f"📊 Endpoint Metrics: {endpoint_name}")
            report_lines.append("-" * 30)
            
            metrics = self.get_endpoint_metrics(endpoint_name)
            
            for metric_name, data in metrics.items():
                if 'error' not in data:
                    report_lines.append(f"  • {metric_name}: {data['latest_value']:.2f} {data['unit']}")
                    report_lines.append(f"    └─ 24h Average: {data['average']:.2f}")
                else:
                    report_lines.append(f"  • {metric_name}: Error - {data['error']}")
            
            report_lines.append("")
        
        # Training job logs
        if training_job_name:
            report_lines.append(f"📝 Training Job Logs: {training_job_name}")
            report_lines.append("-" * 30)
            
            logs = self.get_training_job_logs(training_job_name, lines=10)
            for log_line in logs[-10:]:  # Last 10 lines
                report_lines.append(f"  {log_line}")
            
            report_lines.append("")
        
        # Cost information
        report_lines.append("💰 Cost Analysis (Last 7 days)")
        report_lines.append("-" * 30)
        
        cost_data = self.get_cost_metrics(days_back=7)
        if 'error' not in cost_data:
            report_lines.append(f"  • Total Cost: ${cost_data['total_cost']:.2f}")
            report_lines.append(f"  • Daily Average: ${cost_data['daily_average']:.2f}")
            
            for service, data in cost_data.get('services', {}).items():
                report_lines.append(f"  • {service}: ${data['total']:.2f}")
        else:
            report_lines.append(f"  • Error retrieving cost data: {cost_data['error']}")
        
        return "\n".join(report_lines)


class InferenceMetricsCollector:
    """Collect custom metrics during inference"""
    
    def __init__(self, monitoring: SageMakerMonitoring):
        self.monitoring = monitoring
    
    def record_inference(self, model_name: str, latency_ms: float, 
                        success: bool = True, input_tokens: int = 0,
                        output_tokens: int = 0):
        """Record metrics for a single inference"""
        
        # Publish latency metric
        self.monitoring.publish_custom_metric(
            'InferenceLatency',
            latency_ms,
            dimensions={'Model': model_name},
            unit='Milliseconds'
        )
        
        # Publish success/error metric
        self.monitoring.publish_custom_metric(
            'InferenceCount',
            1,
            dimensions={'Model': model_name, 'Status': 'Success' if success else 'Error'},
            unit='Count'
        )
        
        # Publish token metrics if provided
        if input_tokens > 0:
            self.monitoring.publish_custom_metric(
                'InputTokens',
                input_tokens,
                dimensions={'Model': model_name},
                unit='Count'
            )
        
        if output_tokens > 0:
            self.monitoring.publish_custom_metric(
                'OutputTokens',
                output_tokens,
                dimensions={'Model': model_name},
                unit='Count'
            )


# Convenience functions
def setup_monitoring_for_endpoint(endpoint_name: str, 
                                config: SageMakerConfigManager = None) -> SageMakerMonitoring:
    """Quick setup monitoring for an endpoint"""
    
    monitoring = SageMakerMonitoring(config)
    
    # Create alarms
    alarms = monitoring.create_alarms(endpoint_name)
    logger.info(f"✅ Created {len(alarms)} alarms for endpoint {endpoint_name}")
    
    return monitoring


def create_project_dashboard(config: SageMakerConfigManager = None) -> str:
    """Create a project-wide monitoring dashboard"""
    
    monitoring = SageMakerMonitoring(config)
    dashboard_url = monitoring.create_dashboard("StrangerThings-SageMaker-Dashboard")
    
    return dashboard_url


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker Monitoring Tools")
    parser.add_argument("--action", choices=["dashboard", "alarms", "report", "metrics"], required=True)
    parser.add_argument("--endpoint-name", help="Endpoint name for monitoring")
    parser.add_argument("--training-job-name", help="Training job name for logs")
    
    args = parser.parse_args()
    
    monitoring = SageMakerMonitoring()
    
    if args.action == "dashboard":
        url = create_project_dashboard()
        print(f"✅ Dashboard created: {url}")
    
    elif args.action == "alarms" and args.endpoint_name:
        alarms = setup_monitoring_for_endpoint(args.endpoint_name)
        print(f"✅ Monitoring setup complete for {args.endpoint_name}")
    
    elif args.action == "report":
        report = monitoring.generate_monitoring_report(
            endpoint_name=args.endpoint_name,
            training_job_name=args.training_job_name
        )
        print(report)
    
    elif args.action == "metrics" and args.endpoint_name:
        metrics = monitoring.get_endpoint_metrics(args.endpoint_name)
        print(f"📊 Metrics for {args.endpoint_name}:")
        for metric, data in metrics.items():
            if 'error' not in data:
                print(f"  {metric}: {data['latest_value']} {data['unit']}")
    
    else:
        parser.print_help()