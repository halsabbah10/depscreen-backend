"""
Dashboard router package.

Assembles sub-routers from focused sub-modules into the single `router`
object that `routes/__init__.py` imports as `dashboard_router`.
"""

from fastapi import APIRouter

from .appointments import router as appointments_router
from .care_plans import router as care_plans_router
from .clinical import router as clinical_router
from .messaging import router as messaging_router
from .scheduling import router as scheduling_router
from .screenings import router as screenings_router
from .stats import router as stats_router

router = APIRouter()
router.include_router(stats_router)
router.include_router(screenings_router)
router.include_router(appointments_router)
router.include_router(care_plans_router)
router.include_router(clinical_router)
router.include_router(scheduling_router)
router.include_router(messaging_router)
