"""
Alert Monitor - Condition-Based Alert Scheduling
==================================================

Monitors financial conditions and triggers alerts:
- Price alerts (above/below thresholds)
- News alerts (keyword triggers)
- Portfolio alerts (allocation drift, risk breaches)
- Market alerts (volatility, volume spikes)

Author: MiniMax Agent
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import time
import json
import logging

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Alert types"""
    PRICE = "price"               # Price threshold alerts
    NEWS = "news"                # News keyword alerts
    PORTFOLIO = "portfolio"      # Portfolio condition alerts
    RISK = "risk"                # Risk metric alerts
    VOLUME = "volume"            # Volume spike alerts
    VOLATILITY = "volatility"     # Volatility alerts
    CUSTOM = "custom"           # Custom condition alerts


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AlertCondition:
    """Condition for alert triggering"""
    field: str                           # Field to check (e.g., "price", "volume")
    operator: str                        # Comparison operator (>, <, ==, >=, <=, contains)
    value: Any                           # Threshold value
    comparison_type: str = "absolute"     # absolute, percentage, historical

    def evaluate(self, data: Dict) -> bool:
        """Evaluate if condition is met"""
        field_value = data.get(self.field)

        if field_value is None:
            return False

        try:
            if self.operator == ">":
                return float(field_value) > float(self.value)
            elif self.operator == "<":
                return float(field_value) < float(self.value)
            elif self.operator == ">=":
                return float(field_value) >= float(self.value)
            elif self.operator == "<=":
                return float(field_value) <= float(self.value)
            elif self.operator == "==":
                return float(field_value) == float(self.value)
            elif self.operator == "contains":
                return str(self.value).lower() in str(field_value).lower()
            return False
        except (ValueError, TypeError):
            return False


@dataclass
class AlertRule:
    """Rule defining when to trigger an alert"""
    rule_id: str
    name: str
    description: str = ""

    # Conditions
    conditions: List[AlertCondition] = field(default_factory=list)
    condition_logic: str = "AND"  # AND, OR

    # Alert settings
    alert_type: AlertType = AlertType.PRICE
    priority: AlertPriority = AlertPriority.NORMAL

    # Data source
    data_source: str = ""        # e.g., "stock", "news", "portfolio"
    data_source_id: str = ""     # e.g., "NVDA", "news_api"

    # Timing
    check_interval_seconds: int = 60  # How often to check
    active_hours: Optional[str] = None  # e.g., "9:00-16:00" for market hours

    # Cooldown (prevent duplicate alerts)
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None

    # Notification
    recipients: List[str] = field(default_factory=list)
    message_template: str = "{alert_name}: {condition_value} reached {threshold}"

    # State
    enabled: bool = True
    trigger_count: int = 0

    def should_trigger(self, data: Dict) -> bool:
        """Check if alert should trigger based on conditions"""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False

        # Evaluate conditions
        results = [condition.evaluate(data) for condition in self.conditions]

        if self.condition_logic == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def trigger(self, data: Dict) -> Dict:
        """Trigger alert and return notification data"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1

        # Format message
        message = self.message_template
        for condition in self.conditions:
            message = message.replace(f"{{{condition.field}}}", str(data.get(condition.field, "N/A")))
            message = message.replace("{threshold}", str(condition.value))
            message = message.replace("{alert_name}", self.name)

        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "alert_type": self.alert_type.value,
            "priority": self.priority.value,
            "message": message,
            "data": data,
            "triggered_at": datetime.now().isoformat(),
            "recipients": self.recipients
        }


@dataclass
class AlertEvent:
    """Triggered alert event"""
    rule_id: str
    rule_name: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    triggered_at: datetime
    data: Dict = field(default_factory=dict)
    sent: bool = False
    delivery_results: Dict = field(default_factory=dict)


class AlertMonitor:
    """
    Monitor for financial conditions and alert triggering.

    Features:
    - Multiple alert types (price, news, portfolio, risk)
    - Flexible condition evaluation
    - Cooldown management
    - Multi-channel notification
    - Alert history

    Example:
        monitor = AlertMonitor()

        # Add price alert
        monitor.add_rule(
            name="NVDA Price Alert",
            alert_type=AlertType.PRICE,
            conditions=[
                AlertCondition(field="price", operator=">", value=900)
            ],
            recipients=["telegram:123456"]
        )

        # Start monitoring
        monitor.start()
    """

    def __init__(self, notification_callback: Optional[Callable] = None):
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[AlertEvent] = []
        self.max_history_size = 1000

        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Data sources (would be connected to actual data in production)
        self.data_sources: Dict[str, Callable] = {}

        # Notification callback
        self.notification_callback = notification_callback

        # Callbacks
        self.on_alert_triggered: Optional[Callable] = None
        self.on_alert_sent: Optional[Callable] = None

        logger.info("AlertMonitor initialized")

    def add_rule(self, **kwargs) -> AlertRule:
        """Add a new alert rule"""
        rule_id = kwargs.get("rule_id", f"rule_{len(self.rules)}_{datetime.now().strftime('%Y%m%d%H%M%S')}")

        rule = AlertRule(
            rule_id=rule_id,
            name=kwargs.get("name", "Unnamed Rule"),
            description=kwargs.get("description", ""),
            alert_type=kwargs.get("alert_type", AlertType.PRICE),
            priority=kwargs.get("priority", AlertPriority.NORMAL),
            conditions=kwargs.get("conditions", []),
            condition_logic=kwargs.get("condition_logic", "AND"),
            data_source=kwargs.get("data_source", ""),
            data_source_id=kwargs.get("data_source_id", ""),
            check_interval_seconds=kwargs.get("check_interval_seconds", 60),
            cooldown_minutes=kwargs.get("cooldown_minutes", 30),
            recipients=kwargs.get("recipients", []),
            message_template=kwargs.get("message_template", "{alert_name} triggered"),
            enabled=kwargs.get("enabled", True)
        )

        self.rules[rule_id] = rule
        logger.info(f"Added alert rule: {rule_id}")
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule"""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule"""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def register_data_source(self, source_name: str, fetch_function: Callable):
        """Register a data source for alert checking"""
        self.data_sources[source_name] = fetch_function
        logger.info(f"Registered data source: {source_name}")

    def _fetch_data(self, data_source: str, data_source_id: str) -> Dict:
        """Fetch data from registered source"""
        if data_source not in self.data_sources:
            # Return mock data if no source registered
            return self._get_mock_data(data_source, data_source_id)

        try:
            return self.data_sources[data_source](data_source_id)
        except Exception as e:
            logger.error(f"Data fetch error for {data_source}: {e}")
            return {}

    def _get_mock_data(self, data_source: str, data_source_id: str) -> Dict:
        """Get mock data for testing"""
        if data_source == "stock":
            return {
                "symbol": data_source_id,
                "price": 100.0 + (time.time() % 10),
                "change": 0.5,
                "volume": 1000000
            }
        elif data_source == "news":
            return {
                "headlines": ["Breaking news", "Market update"],
                "keywords_matched": []
            }
        elif data_source == "portfolio":
            return {
                "total_value": 100000,
                "tech_allocation": 65,
                "var": 5000
            }
        return {}

    def _check_rule(self, rule: AlertRule) -> Optional[AlertEvent]:
        """Check a single rule and trigger if conditions met"""
        # Fetch data
        data = self._fetch_data(rule.data_source, rule.data_source_id)

        if not data:
            return None

        # Check conditions
        if rule.should_trigger(data):
            # Trigger alert
            notification_data = rule.trigger(data)

            event = AlertEvent(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                alert_type=rule.alert_type,
                priority=rule.priority,
                message=notification_data["message"],
                triggered_at=datetime.now(),
                data=data
            )

            return event

        return None

    def _send_notification(self, event: AlertEvent):
        """Send alert notification"""
        from platforms import SocialPlatformManager, Platform

        manager = SocialPlatformManager()
        manager.initialize()

        for recipient in event.data.get("recipients", []):
            if ":" in recipient:
                platform_str, recipient_id = recipient.split(":", 1)
                try:
                    platform = Platform(platform_str.lower())

                    result = manager.send_alert(
                        platform=platform,
                        recipient=recipient_id,
                        alert_type=event.alert_type.value,
                        title=event.rule_name,
                        message=event.message,
                        priority=event.priority.value
                    )

                    event.delivery_results[f"{platform_str}:{recipient_id}"] = result.success

                except ValueError:
                    pass

        event.sent = True

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Alert monitoring started")

        while self.running:
            try:
                for rule in list(self.rules.values()):
                    if not rule.enabled:
                        continue

                    event = self._check_rule(rule)

                    if event:
                        # Store event
                        self.alert_history.append(event)
                        if len(self.alert_history) > self.max_history_size:
                            self.alert_history = self.alert_history[-self.max_history_size:]

                        # Send notification
                        self._send_notification(event)

                        # Call callback
                        if self.on_alert_triggered:
                            try:
                                self.on_alert_triggered(event)
                            except Exception as e:
                                logger.warning(f"Alert triggered callback error: {e}")

                # Sleep for check interval
                time.sleep(1)  # Check every second, rules handle their own intervals

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)

    def start(self):
        """Start alert monitoring"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("AlertMonitor started")

    def stop(self):
        """Stop alert monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("AlertMonitor stopped")

    def get_active_rules(self) -> List[AlertRule]:
        """Get all active alert rules"""
        return [r for r in self.rules.values() if r.enabled]

    def get_alert_history(
        self,
        rule_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        limit: int = 100
    ) -> List[AlertEvent]:
        """Get alert history, optionally filtered"""
        history = self.alert_history

        if rule_id:
            history = [e for e in history if e.rule_id == rule_id]

        if alert_type:
            history = [e for e in history if e.alert_type == alert_type]

        return history[-limit:]

    def get_stats(self) -> Dict:
        """Get alert statistics"""
        total = len(self.alert_history)
        sent = sum(1 for e in self.alert_history if e.sent)

        by_type = {}
        for event in self.alert_history:
            type_key = event.alert_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        by_priority = {}
        for event in self.alert_history:
            priority_key = event.priority.value
            by_priority[priority_key] = by_priority.get(priority_key, 0) + 1

        return {
            "total_alerts": total,
            "alerts_sent": sent,
            "active_rules": len(self.get_active_rules()),
            "total_rules": len(self.rules),
            "by_type": by_type,
            "by_priority": by_priority
        }


# Predefined alert templates
ALERT_TEMPLATES = {
    "price_above": {
        "name": "Price Above Threshold",
        "alert_type": AlertType.PRICE,
        "conditions": [AlertCondition(field="price", operator=">", value=0)],
        "message_template": "{alert_name}: Price is now ${price}"
    },
    "price_below": {
        "name": "Price Below Threshold",
        "alert_type": AlertType.PRICE,
        "conditions": [AlertCondition(field="price", operator="<", value=0)],
        "message_template": "{alert_name}: Price dropped to ${price}"
    },
    "price_change_pct": {
        "name": "Price Change Percentage",
        "alert_type": AlertType.PRICE,
        "conditions": [AlertCondition(field="change", operator=">", value=0)],
        "message_template": "{alert_name}: Price changed by {change}%"
    },
    "high_volume": {
        "name": "High Volume Alert",
        "alert_type": AlertType.VOLUME,
        "conditions": [AlertCondition(field="volume", operator=">", value=0)],
        "message_template": "{alert_name}: Unusual volume detected"
    },
    "high_var": {
        "name": "High VaR Alert",
        "alert_type": AlertType.RISK,
        "conditions": [AlertCondition(field="var", operator=">", value=0)],
        "message_template": "{alert_name}: Risk exceeded threshold"
    },
    "tech_allocation_drift": {
        "name": "Tech Allocation Drift",
        "alert_type": AlertType.PORTFOLIO,
        "conditions": [AlertCondition(field="tech_allocation", operator=">", value=70)],
        "message_template": "{alert_name}: Tech allocation is {tech_allocation}%"
    }
}


def create_alert_from_template(
    monitor: AlertMonitor,
    template_name: str,
    **overrides
) -> AlertRule:
    """Create an alert rule from a predefined template"""
    template = ALERT_TEMPLATES.get(template_name)
    if not template:
        raise ValueError(f"Unknown template: {template_name}")

    kwargs = {**template, **overrides}
    return monitor.add_rule(**kwargs)